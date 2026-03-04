import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fdad.config import FDADConfig
from fdad.vocabulary import Vocabulary
from fdad.distribution_router import DistributionAwareProgram
from fdad.soft_program import SoftProgram
from fdad.discretization import (
    DiscretizationScheduler, extract_discrete_program, discreteness_score,
)
from fdad.data import SortingDataGenerator
from fdad.losses import total_loss
from fdad.visualize import print_discrete_program


class FDADTrainer:
    """Multi-phase training loop for algorithm discovery.

    Phase 1 (Exploration): High temperature, learn basic sorting behavior.
    Phase 2 (Refinement/Vocab Expansion): Monitor utility, create compound ops.
    Phase 3 (Discretization): Anneal temperature, collapse to hard program.
    """

    def __init__(self, config=None):
        if config is None:
            config = FDADConfig()
        self.config = config

        # Handle device
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            config.device = 'cpu'
        self.device = config.device

        # Initialize vocabulary
        self.vocab = Vocabulary(beta=config.beta)

        # Initialize model
        if config.num_programs > 1:
            self.model = DistributionAwareProgram(
                num_programs=config.num_programs,
                num_outer_iters=config.num_outer_iters,
                num_inner_steps=config.num_inner_steps,
                vocab=self.vocab,
                max_array_size=config.max_array_size,
                num_registers=config.num_registers,
                max_ptrs=config.max_ptrs,
                max_scalars=config.max_scalars,
            ).to(self.device)
        else:
            self.model = SoftProgram(
                num_outer_iters=config.num_outer_iters,
                num_inner_steps=config.num_inner_steps,
                vocab_size=self.vocab.size,
                max_array_size=config.max_array_size,
                num_registers=config.num_registers,
                max_ptrs=config.max_ptrs,
                max_scalars=config.max_scalars,
            ).to(self.device)

        self.data_gen = SortingDataGenerator()
        self.scheduler = DiscretizationScheduler(
            config.total_steps,
            config.exploration_frac,
            config.refinement_frac,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )
        self.step = 0
        self.history = []

    def train_step(self):
        """Execute a single training step.

        Returns:
            dict of metrics for this step
        """
        self.model.train()

        # Sample array size from training range
        array_size = random.randint(
            self.config.min_array_size, self.config.max_array_size
        )

        # Generate mixed-distribution batch
        arrays, targets, dist_labels = self.data_gen.generate_mixed_batch(
            self.config.batch_size, array_size, self.device
        )

        # Get current schedule parameters
        tau = self.scheduler.get_tau(self.step)
        beta = self.scheduler.get_beta(self.step, self.config.beta)
        entropy_w = self.scheduler.get_entropy_weight(self.step)
        content_w = self.scheduler.get_content_weight(self.step)
        phase = self.scheduler.get_phase(self.step)

        # Forward pass
        if isinstance(self.model, DistributionAwareProgram):
            predicted, route_weights, all_step_info = self.model(
                arrays, tau, beta
            )
        else:
            predicted, step_info = self.model(
                arrays, self.vocab, tau, beta
            )
            route_weights = None
            all_step_info = [step_info]

        # Compute loss
        loss = total_loss(
            predicted, targets, self.model,
            entropy_weight=entropy_w,
            sparsity_weight=0.01,
            ordering_weight=0.1,
            content_weight=content_w,
            beta=beta,
        )

        # Backward + optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        # Collect gradient info for vocabulary expansion (before optimizer step)
        grad_info = self._collect_grad_info()

        self.optimizer.step()
        self.step += 1

        # Update vocabulary utility tracking
        for info in all_step_info:
            self.vocab.update_utility(info, grad_info)

        # Compute metrics
        with torch.no_grad():
            mse = F.mse_loss(predicted, targets).item()
            # Check ordering correctness
            pred_order = predicted.argsort(dim=-1)
            target_order = targets.argsort(dim=-1)
            exact_match = (
                (pred_order == target_order).all(dim=-1).float().mean().item()
            )

        metrics = {
            'loss': loss.item(),
            'mse': mse,
            'exact_match': exact_match,
            'tau': tau,
            'beta': beta,
            'phase': phase,
            'array_size': array_size,
            'vocab_size': self.vocab.size,
        }

        return metrics

    def _collect_grad_info(self):
        """Collect gradient magnitudes for vocabulary expansion."""
        grad_info = {}
        programs = self._get_programs()
        for prog_idx, prog in enumerate(programs):
            for step_idx, step in enumerate(prog.inner_block.steps):
                if step.op_logits.grad is not None:
                    key = (prog_idx, step_idx)
                    grad_info[key] = step.op_logits.grad.abs().detach()
        return grad_info

    def _get_programs(self):
        """Get list of SoftProgram instances from the model."""
        if isinstance(self.model, DistributionAwareProgram):
            return list(self.model.programs)
        return [self.model]

    def maybe_expand_vocabulary(self):
        """Check if vocabulary should be expanded. Run during refinement phase."""
        expanded = False
        for prog in self._get_programs():
            candidates = self.vocab.check_expansion(
                prog, threshold=self.config.expansion_threshold
            )
            for start, end, name in candidates:
                print(f"  Expanding vocabulary: {name} "
                      f"(steps {start}-{end})")
                self.vocab.expand(prog, start, end, name)
                expanded = True

        if expanded:
            # Re-create optimizer to include new parameters
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.lr
            )

    def evaluate(self, array_sizes=None, distributions=None):
        """Evaluate sorting accuracy across sizes and distributions.

        Args:
            array_sizes: list of sizes to evaluate
            distributions: list of distribution names
        Returns:
            dict mapping (size, dist) -> {mse, exact_match}
        """
        if array_sizes is None:
            array_sizes = self.config.ood_sizes
        if distributions is None:
            distributions = SortingDataGenerator.DISTRIBUTIONS

        results = {}
        self.model.eval()

        with torch.no_grad():
            for size in array_sizes:
                for dist in distributions:
                    arrays, targets, _ = self.data_gen.generate_batch(
                        self.config.eval_batch_size, size, dist, self.device
                    )

                    if isinstance(self.model, DistributionAwareProgram):
                        predicted, _, _ = self.model(
                            arrays, tau=0.01,
                            beta=self.config.beta * 3
                        )
                    else:
                        predicted, _ = self.model(
                            arrays, self.vocab,
                            tau=0.01,
                            beta=self.config.beta * 3
                        )

                    mse = F.mse_loss(predicted, targets).item()
                    pred_order = predicted.argsort(dim=-1)
                    target_order = targets.argsort(dim=-1)
                    em = (
                        (pred_order == target_order)
                        .all(dim=-1).float().mean().item()
                    )
                    results[(size, dist)] = {
                        'mse': mse,
                        'exact_match': em,
                    }

        self.model.train()
        return results

    def save_checkpoint(self, step):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f'step_{step}.pt')
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_op_names': self.vocab.op_names,
            'config': self.config,
        }, path)

    def train(self):
        """Full training loop.

        Returns:
            dict of extracted discrete programs
        """
        print(f"Starting FDAD training on {self.device}")
        print(f"  Array sizes: {self.config.min_array_size}-"
              f"{self.config.max_array_size}")
        print(f"  Program: {self.config.num_outer_iters} outer iters x "
              f"{self.config.num_inner_steps} inner steps")
        print(f"  Vocabulary: {self.vocab.size} operations")
        print(f"  Distribution programs: {self.config.num_programs}")
        print(f"  Total steps: {self.config.total_steps}")
        print()

        pbar = tqdm(range(self.config.total_steps), desc="Training")

        for step in pbar:
            metrics = self.train_step()
            self.history.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'em': f"{metrics['exact_match']:.2f}",
                'tau': f"{metrics['tau']:.3f}",
                'phase': metrics['phase'][:4],
            })

            # Periodic vocabulary expansion check (refinement phase only)
            phase = self.scheduler.get_phase(step)
            if (phase == 'refinement' and
                    step % self.config.expansion_check_freq == 0):
                self.maybe_expand_vocabulary()

            # Periodic logging
            if step % self.config.log_freq == 0 and step > 0:
                self._log_step(step, metrics)

            # Periodic evaluation
            if step % self.config.eval_freq == 0 and step > 0:
                results = self.evaluate(
                    array_sizes=[
                        self.config.min_array_size,
                        self.config.max_array_size,
                    ]
                )
                self._log_eval(step, results)

            # Periodic checkpointing
            if (step % self.config.checkpoint_freq == 0 and step > 0):
                self.save_checkpoint(step)

        # Final evaluation and extraction
        print("\n" + "=" * 60)
        print("Training complete. Extracting discrete programs...")
        print("=" * 60)

        discrete_programs = self.extract_all_programs()
        for name, extracted in discrete_programs.items():
            print_discrete_program(extracted, title=name)

        # Final evaluation on all sizes
        print("\nFinal Evaluation:")
        final_results = self.evaluate()
        self._log_eval(self.step, final_results, verbose=True)

        return discrete_programs

    def extract_all_programs(self):
        """Extract discrete programs from all specialized soft programs."""
        programs = {}
        for i, prog in enumerate(self._get_programs()):
            d_score = discreteness_score(prog, self.vocab)
            extracted = extract_discrete_program(prog, self.vocab)
            extracted['discreteness'] = d_score
            programs[f'program_{i}'] = extracted
        return programs

    def _log_step(self, step, metrics):
        """Log training step metrics."""
        phase = metrics['phase']
        tqdm.write(
            f"  Step {step:6d} | phase={phase:12s} | "
            f"loss={metrics['loss']:.4f} | "
            f"mse={metrics['mse']:.4f} | "
            f"EM={metrics['exact_match']:.2%} | "
            f"tau={metrics['tau']:.3f} | "
            f"vocab={metrics['vocab_size']}"
        )

    def _log_eval(self, step, results, verbose=False):
        """Log evaluation results."""
        # Aggregate by size
        size_results = {}
        for (size, dist), vals in results.items():
            if size not in size_results:
                size_results[size] = {'mse': [], 'exact_match': []}
            size_results[size]['mse'].append(vals['mse'])
            size_results[size]['exact_match'].append(vals['exact_match'])

        for size in sorted(size_results.keys()):
            avg_mse = sum(size_results[size]['mse']) / len(
                size_results[size]['mse']
            )
            avg_em = sum(size_results[size]['exact_match']) / len(
                size_results[size]['exact_match']
            )
            ood = " (OOD)" if size > self.config.max_array_size else ""
            tqdm.write(
                f"  Eval step {step:6d} | size={size:2d}{ood} | "
                f"MSE={avg_mse:.4f} | EM={avg_em:.2%}"
            )

        if verbose:
            # Per-distribution breakdown
            for (size, dist), vals in sorted(results.items()):
                ood = " (OOD)" if size > self.config.max_array_size else ""
                tqdm.write(
                    f"    size={size:2d}{ood} dist={dist:15s} | "
                    f"MSE={vals['mse']:.4f} | EM={vals['exact_match']:.2%}"
                )


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description='FDAD Training')
    parser.add_argument('--min_array_size', type=int, default=4)
    parser.add_argument('--max_array_size', type=int, default=16)
    parser.add_argument('--num_outer_iters', type=int, default=16)
    parser.add_argument('--num_inner_steps', type=int, default=8)
    parser.add_argument('--num_programs', type=int, default=4)
    parser.add_argument('--total_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = FDADConfig(
        min_array_size=args.min_array_size,
        max_array_size=args.max_array_size,
        num_outer_iters=args.num_outer_iters,
        num_inner_steps=args.num_inner_steps,
        num_programs=args.num_programs,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        device=args.device,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
    )

    trainer = FDADTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
