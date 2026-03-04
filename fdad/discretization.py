import torch
import torch.nn.functional as F


class DiscretizationScheduler:
    """Manages temperature annealing and entropy regularization weight.

    Three training phases:
    1. Exploration (high tau=1.0): no entropy penalty, free exploration
    2. Refinement (tau: 1.0 -> 0.1): mild entropy penalty, vocab expansion
    3. Collapse (tau: 0.1 -> 0.01): strong entropy penalty, force hard selections
    """

    def __init__(self, total_steps, exploration_frac=0.3, refinement_frac=0.4):
        self.total_steps = total_steps
        self.exploration_frac = exploration_frac
        self.refinement_frac = refinement_frac
        self.exploration_end = int(total_steps * exploration_frac)
        self.refinement_end = int(
            total_steps * (exploration_frac + refinement_frac)
        )

    def get_phase(self, step):
        """Get current training phase name."""
        if step < self.exploration_end:
            return 'exploration'
        elif step < self.refinement_end:
            return 'refinement'
        else:
            return 'collapse'

    def get_tau(self, step):
        """Current Gumbel-Softmax temperature."""
        if step < self.exploration_end:
            return 1.0
        elif step < self.refinement_end:
            progress = (step - self.exploration_end) / max(
                self.refinement_end - self.exploration_end, 1
            )
            # Exponential decay: 1.0 -> 0.1
            return 1.0 * (0.1 / 1.0) ** progress
        else:
            progress = (step - self.refinement_end) / max(
                self.total_steps - self.refinement_end, 1
            )
            # Exponential decay: 0.1 -> 0.01
            return 0.1 * (0.01 / 0.1) ** min(progress, 1.0)

    def get_entropy_weight(self, step):
        """Weight for entropy regularization loss."""
        if step < self.exploration_end:
            return 0.0
        elif step < self.refinement_end:
            progress = (step - self.exploration_end) / max(
                self.refinement_end - self.exploration_end, 1
            )
            return 0.01 * progress
        else:
            return 0.1

    def get_content_weight(self, step):
        """Weight for content preservation loss (ramps up during collapse)."""
        if step < self.refinement_end:
            return 0.0
        progress = (step - self.refinement_end) / max(
            self.total_steps - self.refinement_end, 1
        )
        return 0.1 * min(progress, 1.0)

    def get_beta(self, step, base_beta=10.0):
        """AlgoVision inverse temperature — increases over training."""
        if step < self.exploration_end:
            return base_beta
        progress = (step - self.exploration_end) / max(
            self.total_steps - self.exploration_end, 1
        )
        return base_beta * (1.0 + 2.0 * min(progress, 1.0))

    def get_lr_multiplier(self, step):
        """Learning rate decay multiplier."""
        if step < self.exploration_end:
            return 1.0
        elif step < self.refinement_end:
            return 0.5
        else:
            return 0.1


def extract_discrete_program(program, vocabulary):
    """Convert a soft program to a discrete, human-readable form.

    For each step in the inner block:
    1. argmax of op_logits -> selected operation
    2. argmax of pointer_logits -> selected positions
    3. Skip if selected op is NoOp

    Also reports the effective loop count from halt_logits.

    Args:
        program: SoftProgram instance
        vocabulary: Vocabulary instance
    Returns:
        dict with 'loop_count', 'steps', and 'description'
    """
    V = vocabulary.size

    # Determine effective loop count
    halt_probs = torch.sigmoid(program.halt_logits)
    effective_iters = program.num_outer_iters
    for k in range(program.num_outer_iters):
        if halt_probs[k].item() > 0.5:
            effective_iters = k + 1
            break

    # Extract discrete steps from inner block
    discrete_steps = []
    for step_idx, step in enumerate(program.inner_block.steps):
        logits = step.op_logits[:V]
        probs = F.softmax(logits, dim=-1)
        op_idx = logits.argmax().item()
        op_name = vocabulary.op_names[op_idx]
        confidence = probs[op_idx].item()

        if op_name == "noop":
            continue

        # Get pointer positions
        positions = step.pointer_logits.argmax(dim=-1).tolist()

        discrete_steps.append({
            'step': step_idx,
            'op': op_name,
            'positions': positions,
            'confidence': confidence,
        })

    # Build human-readable description
    lines = [f"REPEAT {effective_iters} times:"]
    for ds in discrete_steps:
        pos_str = ", ".join(f"pos[{p}]" for p in ds['positions'])
        lines.append(
            f"  step {ds['step']}: {ds['op']}({pos_str})  "
            f"[conf={ds['confidence']:.2f}]"
        )

    return {
        'loop_count': effective_iters,
        'steps': discrete_steps,
        'description': "\n".join(lines),
        'halt_probs': halt_probs.detach().tolist(),
    }


def discreteness_score(program, vocabulary):
    """Compute how "discrete" the current program is.

    Returns the average max-probability across all instruction slots.
    A score near 1.0 means the program is effectively discrete.

    Args:
        program: SoftProgram instance
        vocabulary: Vocabulary instance
    Returns:
        float in [0, 1]
    """
    V = vocabulary.size
    scores = []
    for step in program.inner_block.steps:
        probs = F.softmax(step.op_logits[:V], dim=-1)
        scores.append(probs.max().item())
    return sum(scores) / max(len(scores), 1)
