import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque
from fdad.primitives import PrimitiveOp, get_default_ops


class UtilityMetrics:
    """Tracks gradient magnitude, selection frequency, and co-occurrence."""

    def __init__(self, window_size=500):
        self.window_size = window_size
        self.grad_magnitudes = deque(maxlen=window_size)
        self.selection_weights = deque(maxlen=window_size)
        self.selection_counts = 0
        self.total_steps = 0

    def update(self, grad_mag, selection_weight):
        self.grad_magnitudes.append(grad_mag)
        self.selection_weights.append(selection_weight)
        self.total_steps += 1
        if selection_weight > 0.5:
            self.selection_counts += 1

    @property
    def mean_grad(self):
        if not self.grad_magnitudes:
            return 0.0
        return sum(self.grad_magnitudes) / len(self.grad_magnitudes)

    @property
    def mean_selection(self):
        if not self.selection_weights:
            return 0.0
        return sum(self.selection_weights) / len(self.selection_weights)

    @property
    def frequency(self):
        if self.total_steps == 0:
            return 0.0
        return self.selection_counts / self.total_steps


class CompoundOp(PrimitiveOp):
    """A compound operation created by distilling a high-utility sub-graph.

    Wraps a frozen sequence of SoftProgramStep copies that execute
    as a single vocabulary entry.
    """
    num_pointer_args = 0
    num_scalar_args = 0

    def __init__(self, name, sub_steps, parent_vocab):
        super().__init__()
        self.name = name
        self.sub_steps = nn.ModuleList(sub_steps)
        self.parent_vocab = parent_vocab
        # Freeze all parameters
        for param in self.sub_steps.parameters():
            param.requires_grad_(False)

    def forward(self, memory, pointers, scalars, beta):
        for sub_step in self.sub_steps:
            memory = sub_step(
                memory, self.parent_vocab, tau=0.01, beta=beta
            )
        return memory


class Vocabulary(nn.Module):
    """Manages the set of available operations and tracks their utility.

    Supports dynamic expansion: when operation sub-sequences show high
    utility (gradient magnitude, frequency, co-occurrence), they can be
    distilled into new CompoundOp entries.
    """

    def __init__(self, beta=10.0):
        super().__init__()
        self.ops = nn.ModuleList()
        self.op_names = []
        self.utility = {}
        self.co_occurrence = None  # (V, V) matrix, lazily initialized
        self._co_occurrence_history = deque(maxlen=500)
        self._register_initial_ops()

    def _register_initial_ops(self):
        for op in get_default_ops():
            self.register_op(op)

    def register_op(self, op):
        """Add an operation to the vocabulary."""
        self.ops.append(op)
        self.op_names.append(op.name)
        self.utility[op.name] = UtilityMetrics()
        # Reset co-occurrence matrix (size changed)
        self.co_occurrence = None

    @property
    def size(self):
        return len(self.ops)

    def __len__(self):
        return len(self.ops)

    def update_utility(self, step_info, grad_info=None):
        """Update utility metrics from a training step.

        Args:
            step_info: dict with 'op_weights' list of (V,) tensors
            grad_info: optional dict mapping (prog_idx, step_idx) to grad tensors
        """
        op_weights_list = step_info.get('op_weights', [])
        V = self.size

        # Update per-op metrics
        for op_weights in op_weights_list:
            if op_weights is None or len(op_weights) < V:
                continue
            for i in range(V):
                w = op_weights[i].item() if isinstance(op_weights[i], torch.Tensor) else op_weights[i]
                grad_mag = 0.0
                if grad_info:
                    # Sum relevant gradients
                    for key, grad in grad_info.items():
                        if len(grad) > i:
                            grad_mag += grad[i].item()
                self.utility[self.op_names[i]].update(grad_mag, w)

        # Update co-occurrence for consecutive steps
        for t in range(len(op_weights_list) - 1):
            w_t = op_weights_list[t]
            w_t1 = op_weights_list[t + 1]
            if w_t is None or w_t1 is None:
                continue
            if len(w_t) >= V and len(w_t1) >= V:
                # Outer product of selection weights
                co = w_t[:V].unsqueeze(-1) * w_t1[:V].unsqueeze(0)
                self._co_occurrence_history.append(co.detach())

    def get_co_occurrence_matrix(self):
        """Compute averaged co-occurrence matrix from history."""
        if not self._co_occurrence_history:
            return torch.zeros(self.size, self.size)
        stacked = torch.stack(list(self._co_occurrence_history), dim=0)
        return stacked.mean(dim=0)

    def check_expansion(self, program, threshold=0.8):
        """Check if any sub-sequences should be promoted to compound ops.

        Criteria for promotion:
        1. A contiguous sub-sequence of 2-5 steps in the inner block
        2. Each step has a dominant operation (max weight > threshold)
        3. The pattern is stable (appears consistently in recent history)

        Args:
            program: SoftProgram instance
            threshold: minimum dominant op weight

        Returns:
            list of (start_step, end_step, proposed_name) tuples
        """
        if not hasattr(program, 'inner_block'):
            return []

        steps = list(program.inner_block.steps)
        candidates = []
        V = self.size

        # Check each pair of consecutive steps
        for start in range(len(steps) - 1):
            # Check if both steps have dominant operations
            dominant_ops = []
            all_dominant = True
            for t in range(start, min(start + 3, len(steps))):
                probs = F.softmax(steps[t].op_logits[:V], dim=-1)
                max_prob = probs.max().item()
                max_idx = probs.argmax().item()
                if max_prob < threshold:
                    all_dominant = False
                    break
                # Skip if it's NoOp
                if self.op_names[max_idx] == "noop":
                    all_dominant = False
                    break
                dominant_ops.append(self.op_names[max_idx])

            if all_dominant and len(dominant_ops) >= 2:
                name = "compound_" + "_".join(dominant_ops)
                # Don't create duplicates
                if name not in self.op_names:
                    end = start + len(dominant_ops)
                    candidates.append((start, end, name))
                    break  # Only one expansion per check

        return candidates

    def expand(self, program, start_step, end_step, name):
        """Distill steps[start:end] into a new CompoundOp.

        Args:
            program: SoftProgram with inner_block
            start_step: first step index to distill
            end_step: last step index (exclusive)
            name: name for the new compound operation
        """
        inner_steps = list(program.inner_block.steps)
        sub_steps = [copy.deepcopy(inner_steps[i])
                     for i in range(start_step, end_step)]

        compound = CompoundOp(name, sub_steps, self)
        self.register_op(compound)

        # Expand op_logits in all program steps to accommodate new vocab entry
        new_V = self.size
        for step in inner_steps:
            old_logits = step.op_logits.data
            old_V = old_logits.shape[0]
            if old_V < new_V:
                new_logits = torch.zeros(new_V, device=old_logits.device)
                new_logits[:old_V] = old_logits
                # Initialize new entry at mean (neutral prior)
                new_logits[old_V:] = old_logits.mean()
                step.op_logits = nn.Parameter(new_logits)

        return compound
