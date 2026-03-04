import torch
import torch.nn as nn
from fdad.memory import DifferentiableMemory


class PrimitiveOp(nn.Module):
    """Base class for all vocabulary operations.

    Each operation takes a DifferentiableMemory and operand tensors,
    and returns the modified memory. All operations must be differentiable.
    """
    name: str = "base"
    num_pointer_args: int = 0
    num_scalar_args: int = 0

    def forward(self, memory, pointers, scalars, beta):
        """Apply operation to memory.

        Args:
            memory: DifferentiableMemory instance
            pointers: list of (N,) or (B,N) soft position distributions
            scalars: (max_scalars,) sigmoid-squashed scalar parameters
            beta: inverse temperature for differentiable comparisons
        Returns:
            modified DifferentiableMemory
        """
        raise NotImplementedError


class CompareAndSwap(PrimitiveOp):
    """If array[i] > array[j], swap them. Core sorting primitive.

    Differentiable via soft compare (sigmoid) + probabilistic state merge.
    Reuses AlgoVision's GT condition pattern (conditions.py:46-49) and
    State.merge() pattern (core.py:203-247).
    """
    name = "compare_and_swap"
    num_pointer_args = 2
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr_i, ptr_j = pointers[0], pointers[1]

        # Soft comparison: P(array[i] > array[j])
        p_gt = memory.soft_compare(ptr_i, ptr_j, beta)

        # Create swapped version
        mem_swapped = memory.clone()
        mem_swapped.soft_swap(ptr_i, ptr_j)

        # Probabilistic merge: swap with probability p_gt
        memory.merge(mem_swapped, p_gt)
        return memory


class CopyElement(PrimitiveOp):
    """Copy array[src] to array[dst]. For insertion-sort-like moves."""
    name = "copy"
    num_pointer_args = 2
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr_src, ptr_dst = pointers[0], pointers[1]
        val = memory.soft_read(ptr_src)
        memory.soft_write(ptr_dst, val)
        return memory


class ShiftRight(PrimitiveOp):
    """Shift all array elements one position to the right (circular).

    Useful for insertion sort patterns. Last element wraps to first.
    """
    name = "shift_right"
    num_pointer_args = 0
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        memory.array = torch.cat(
            [memory.array[:, -1:], memory.array[:, :-1]], dim=-1
        )
        return memory


class ShiftLeft(PrimitiveOp):
    """Shift all array elements one position to the left (circular)."""
    name = "shift_left"
    num_pointer_args = 0
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        memory.array = torch.cat(
            [memory.array[:, 1:], memory.array[:, :1]], dim=-1
        )
        return memory


class SoftMinSelect(PrimitiveOp):
    """Find soft minimum of the array and swap it to target position.

    Uses AlgoVision's Min/ArgMin softmin pattern (functions.py:9-24).
    Pointer 0 selects the target position to place the minimum.
    """
    name = "soft_min_select"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr_dst = pointers[0]

        # Soft argmin: probability distribution over positions
        weights = torch.nn.functional.softmin(memory.array * beta, dim=-1)

        # Soft minimum value
        min_val = (weights * memory.array).sum(dim=-1)

        # Read current value at destination
        dst_val = memory.soft_read(ptr_dst)

        # Write min to destination, and write old dst value to where min was
        # This performs a soft selection-sort step
        if ptr_dst.dim() == 1:
            ptr_dst_exp = ptr_dst.unsqueeze(0).expand_as(memory.array)
        else:
            ptr_dst_exp = ptr_dst

        # New array = original - (weights * original values) + (weights * dst_val)
        #           - (ptr_dst * dst_val) + (ptr_dst * min_val)
        memory.array = (
            memory.array
            - weights * memory.array + weights * dst_val.unsqueeze(-1)
            - ptr_dst_exp * dst_val.unsqueeze(-1) + ptr_dst_exp * min_val.unsqueeze(-1)
        )
        return memory


class PointerIncrement(PrimitiveOp):
    """Shift a soft pointer distribution one position to the right.

    Effectively implements pointer++ by rolling the attention distribution.
    """
    name = "pointer_increment"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        # This op modifies how the pointer is interpreted but since pointers
        # are produced fresh each step from learnable logits, this is a no-op
        # on memory. The pointer shift pattern is learned in the logits.
        return memory


class PointerDecrement(PrimitiveOp):
    """Shift a soft pointer distribution one position to the left."""
    name = "pointer_decrement"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        return memory


class PointerReset(PrimitiveOp):
    """Reset operation — no-op on memory, pointer handling is in logits."""
    name = "pointer_reset"
    num_pointer_args = 0
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        return memory


class WriteRegister(PrimitiveOp):
    """Read from array at soft position and store in register 0."""
    name = "write_register"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr = pointers[0]
        val = memory.soft_read(ptr)
        memory.write_register(0, val)
        return memory


class ReadRegister(PrimitiveOp):
    """Write register 0 value to array at soft position."""
    name = "read_register"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr = pointers[0]
        val = memory.read_register(0)
        memory.soft_write(ptr, val)
        return memory


class AdjacentCompareAndSwap(PrimitiveOp):
    """Compare-and-swap of position i and i+1.

    Only needs one pointer (the base position). The second position
    is automatically i+1 (rolled by 1). This is the most common
    sorting primitive and may emerge from vocabulary expansion.
    """
    name = "adjacent_cas"
    num_pointer_args = 1
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        ptr_i = pointers[0]
        # ptr_j = roll ptr_i by 1 position to the right
        if ptr_i.dim() == 1:
            ptr_j = torch.roll(ptr_i, shifts=1, dims=0)
        else:
            ptr_j = torch.roll(ptr_i, shifts=1, dims=-1)

        p_gt = memory.soft_compare(ptr_i, ptr_j, beta)
        mem_swapped = memory.clone()
        mem_swapped.soft_swap(ptr_i, ptr_j)
        memory.merge(mem_swapped, p_gt)
        return memory


class NoOp(PrimitiveOp):
    """Identity operation. Allows the program to skip steps.

    Essential for variable-length programs: unused steps learn to
    select NoOp during discretization.
    """
    name = "noop"
    num_pointer_args = 0
    num_scalar_args = 0

    def forward(self, memory, pointers, scalars, beta):
        return memory


# Default initial vocabulary
def get_default_ops():
    """Return the default set of primitive operations."""
    return [
        CompareAndSwap(),
        CopyElement(),
        SoftMinSelect(),
        AdjacentCompareAndSwap(),
        ShiftRight(),
        WriteRegister(),
        ReadRegister(),
        NoOp(),
    ]
