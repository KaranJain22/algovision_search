import torch
import torch.nn as nn
import torch.nn.functional as F
from fdad.memory import DifferentiableMemory


class SoftProgramStep(nn.Module):
    """One instruction slot in the soft program.

    Learnable parameters:
        - op_logits: (V,) which operation to apply
        - pointer_logits: (max_ptrs, N) which array positions to use as operands
        - scalar_params: (max_scalars,) scalar parameters for ops that need them

    At each forward pass:
        1. Gumbel-Softmax over op_logits -> op weights (V,)
        2. Softmax over pointer_logits -> soft positions (max_ptrs, N)
        3. Execute all V ops on cloned memories, blend results by weights
    """

    def __init__(self, vocab_size, max_array_size, max_ptrs=2, max_scalars=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_array_size = max_array_size
        self.max_ptrs = max_ptrs
        self.max_scalars = max_scalars

        self.op_logits = nn.Parameter(torch.randn(vocab_size) * 0.1)
        self.pointer_logits = nn.Parameter(torch.randn(max_ptrs, max_array_size) * 0.1)
        self.scalar_params = nn.Parameter(torch.randn(max_scalars) * 0.1)

    def forward(self, memory, vocabulary, tau, beta, array_size=None):
        """Execute one soft program step.

        Args:
            memory: DifferentiableMemory instance
            vocabulary: Vocabulary with .ops list
            tau: Gumbel-Softmax temperature
            beta: inverse temperature for comparisons
            array_size: actual array size (for masking if < max_array_size)
        Returns:
            updated DifferentiableMemory
        """
        V = len(vocabulary.ops)

        # Operation selection via Gumbel-Softmax
        if self.training:
            op_weights = F.gumbel_softmax(
                self.op_logits[:V], tau=tau, hard=False
            )
        else:
            op_weights = F.softmax(self.op_logits[:V] / max(tau, 1e-6), dim=-1)

        # Pointer selection: softmax over valid positions
        N = array_size if array_size is not None else self.max_array_size
        ptr_logits = self.pointer_logits[:, :N]
        ptrs = F.softmax(ptr_logits / max(tau, 1e-6), dim=-1)  # (max_ptrs, N)
        scalars = torch.sigmoid(self.scalar_params)  # (max_scalars,)

        # Execute all operations and blend results
        result_arrays = []
        result_registers = []

        for op_idx, op in enumerate(vocabulary.ops):
            mem_copy = memory.clone()
            op_ptrs = [ptrs[j] for j in range(min(op.num_pointer_args, self.max_ptrs))]
            op_scalars = scalars[:op.num_scalar_args]
            result = op(mem_copy, op_ptrs, op_scalars, beta)
            result_arrays.append(result.array)
            result_registers.append(result.registers)

        # Weighted merge: sum_k(w_k * result_k)
        stacked_arrays = torch.stack(result_arrays, dim=0)      # (V, B, N)
        stacked_regs = torch.stack(result_registers, dim=0)     # (V, B, R)
        w = op_weights.view(-1, 1, 1)                           # (V, 1, 1)

        memory.array = (w * stacked_arrays).sum(dim=0)          # (B, N)
        memory.registers = (w * stacked_regs).sum(dim=0)        # (B, R)
        return memory


class SoftProgramBlock(nn.Module):
    """Inner block of T instruction steps, executed sequentially."""

    def __init__(self, num_steps, vocab_size, max_array_size,
                 max_ptrs=2, max_scalars=2):
        super().__init__()
        self.steps = nn.ModuleList([
            SoftProgramStep(vocab_size, max_array_size, max_ptrs, max_scalars)
            for _ in range(num_steps)
        ])

    def forward(self, memory, vocabulary, tau, beta, array_size=None):
        for step in self.steps:
            memory = step(memory, vocabulary, tau, beta, array_size)
        return memory


class SoftProgram(nn.Module):
    """A full soft program: K outer iterations of a SHARED T-step inner block.

    Architecture mirrors For(While(...)) from AlgoVision bubble sort.
    Weight-tying across outer iterations is critical:
    - Forces discovery of an actual loop body (not K*T independent steps)
    - Dramatically reduces parameters
    - Mirrors real sorting algorithms where the same logic repeats
    - Halting mechanism allows variable effective loop count

    The halting mechanism mirrors AlgoVision's While loop probabilistic
    accumulation (control_structures.py:82-120).
    """

    def __init__(self, num_outer_iters, num_inner_steps, vocab_size,
                 max_array_size, num_registers=4, max_ptrs=2, max_scalars=2):
        super().__init__()
        self.num_outer_iters = num_outer_iters
        self.max_array_size = max_array_size
        self.num_registers = num_registers

        # Shared inner block (weight-tied across outer iterations)
        self.inner_block = SoftProgramBlock(
            num_inner_steps, vocab_size, max_array_size, max_ptrs, max_scalars
        )

        # Learnable halting: soft "done" probability per outer iteration
        self.halt_logits = nn.Parameter(torch.zeros(num_outer_iters))

    def forward(self, input_array, vocabulary, tau, beta):
        """Execute the full program.

        Args:
            input_array: (B, N) unsorted array
            vocabulary: Vocabulary instance
            tau: Gumbel-Softmax temperature
            beta: inverse temperature
        Returns:
            output_array: (B, N) sorted result
            step_info: dict with per-step diagnostics
        """
        B, N = input_array.shape
        memory = DifferentiableMemory(input_array, self.num_registers)

        # Probabilistic halting accumulation
        halt_probs = torch.sigmoid(self.halt_logits)
        cumulative_continue = torch.ones(B, device=input_array.device)
        accumulated_output = torch.zeros_like(input_array)

        step_info = {
            'op_weights': [],
            'pointer_positions': [],
            'halt_probs': halt_probs.detach().clone(),
        }

        for k in range(self.num_outer_iters):
            memory = self.inner_block(memory, vocabulary, tau, beta,
                                       array_size=N)

            # Record diagnostics (detached, no gradient impact)
            with torch.no_grad():
                for step in self.inner_block.steps:
                    step_info['op_weights'].append(
                        F.softmax(step.op_logits[:len(vocabulary.ops)],
                                  dim=-1).detach()
                    )
                    step_info['pointer_positions'].append(
                        F.softmax(step.pointer_logits[:, :N],
                                  dim=-1).detach()
                    )

            # Soft halting accumulation (mirrors While's accumulate pattern)
            p_halt = halt_probs[k]
            accumulated_output = (
                accumulated_output
                + cumulative_continue.unsqueeze(-1) * p_halt * memory.array
            )
            cumulative_continue = cumulative_continue * (1.0 - p_halt)

        # Final residual: whatever probability mass remains
        accumulated_output = (
            accumulated_output
            + cumulative_continue.unsqueeze(-1) * memory.array
        )

        return accumulated_output, step_info

    def get_effective_iters(self):
        """Number of outer iterations before halting (discrete estimate)."""
        halt_probs = torch.sigmoid(self.halt_logits)
        # Find first iteration where halt prob > 0.5
        for k in range(self.num_outer_iters):
            if halt_probs[k].item() > 0.5:
                return k + 1
        return self.num_outer_iters
