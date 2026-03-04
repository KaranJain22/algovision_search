import torch
import torch.nn as nn
import copy


class DifferentiableMemory:
    """Differentiable memory for soft programs.

    Holds the main array being sorted plus auxiliary registers.
    All reads/writes use soft (differentiable) operations.

    Attributes:
        array: (B, N) the main array being sorted
        registers: (B, R) auxiliary scalar registers for temp values
    """

    def __init__(self, array, num_registers=4):
        """
        Args:
            array: (B, N) tensor or raw input array to initialize from
            num_registers: number of auxiliary registers
        """
        if isinstance(array, torch.Tensor):
            self.array = array.clone()
            self.registers = torch.zeros(
                array.shape[0], num_registers,
                device=array.device, dtype=array.dtype,
            )
        else:
            raise TypeError(f"Expected torch.Tensor, got {type(array)}")

    def soft_read(self, pointer):
        """Read from array using soft pointer (attention distribution).

        Args:
            pointer: (N,) or (B, N) softmax distribution over positions
        Returns:
            (B,) weighted sum of array elements
        """
        if pointer.dim() == 1:
            pointer = pointer.unsqueeze(0).expand_as(self.array)
        return (self.array * pointer).sum(dim=-1)

    def soft_write(self, pointer, value):
        """Write value to array at soft position via probabilistic update.

        Implements: array_new = pointer * value + (1 - pointer) * array_old
        This mirrors State.probabilistic_update() from algovision/core.py:249-258.

        Args:
            pointer: (N,) or (B, N) softmax distribution over positions
            value: (B,) value to write
        """
        if pointer.dim() == 1:
            pointer = pointer.unsqueeze(0).expand_as(self.array)
        self.array = pointer * value.unsqueeze(-1) + (1 - pointer) * self.array

    def soft_swap(self, ptr_i, ptr_j):
        """Differentiable swap: read both positions, write each to the other.

        Must read both values before writing either, to avoid corruption.

        Args:
            ptr_i: (N,) or (B, N) soft pointer to first position
            ptr_j: (N,) or (B, N) soft pointer to second position
        """
        val_i = self.soft_read(ptr_i)
        val_j = self.soft_read(ptr_j)
        # Write in terms of the original array to avoid sequential corruption
        if ptr_i.dim() == 1:
            ptr_i = ptr_i.unsqueeze(0).expand_as(self.array)
        if ptr_j.dim() == 1:
            ptr_j = ptr_j.unsqueeze(0).expand_as(self.array)
        # Compute new array in one shot
        self.array = (
            self.array
            - ptr_i * (self.soft_read_raw(ptr_i) - val_j).unsqueeze(-1)
            - ptr_j * (self.soft_read_raw(ptr_j) - val_i).unsqueeze(-1)
        )

    def soft_read_raw(self, pointer):
        """Read without expanding pointer (helper for swap)."""
        return (self.array * pointer).sum(dim=-1)

    def soft_compare(self, ptr_i, ptr_j, beta):
        """Soft GT comparison using sigmoid relaxation.

        Reuses the pattern from algovision/conditions.py:GT (line 46-49).

        Args:
            ptr_i: soft pointer to first position
            ptr_j: soft pointer to second position
            beta: inverse temperature for sigmoid
        Returns:
            (B,) probability that array[i] > array[j]
        """
        val_i = self.soft_read(ptr_i)
        val_j = self.soft_read(ptr_j)
        return torch.sigmoid((val_i - val_j) * beta)

    def read_register(self, reg_idx):
        """Read a register value.

        Args:
            reg_idx: int or (B,) soft index via softmax over registers
        Returns:
            (B,) register value
        """
        if isinstance(reg_idx, int):
            return self.registers[:, reg_idx]
        # Soft register read
        return (self.registers * reg_idx.unsqueeze(0) if reg_idx.dim() == 1
                else (self.registers * reg_idx).sum(dim=-1))

    def write_register(self, reg_idx, value):
        """Write a value to a register.

        Args:
            reg_idx: int index
            value: (B,) value to write
        """
        if isinstance(reg_idx, int):
            self.registers[:, reg_idx] = value
        else:
            # Soft register write
            if reg_idx.dim() == 1:
                reg_idx = reg_idx.unsqueeze(0).expand_as(self.registers)
            self.registers = (
                reg_idx * value.unsqueeze(-1)
                + (1 - reg_idx) * self.registers
            )

    def merge(self, other, p):
        """Probabilistic merge with another memory state.

        Mirrors State.merge() from algovision/core.py:203-247.

        Args:
            other: DifferentiableMemory to merge from
            p: (B,) probability of using `other`'s values
        """
        p_expand = p.unsqueeze(-1)
        self.array = (1 - p_expand) * self.array + p_expand * other.array
        self.registers = (1 - p_expand) * self.registers + p_expand * other.registers

    def clone(self):
        """Deep copy for branching (mirrors State.clone() from core.py:301)."""
        new = DifferentiableMemory.__new__(DifferentiableMemory)
        new.array = self.array.clone()
        new.registers = self.registers.clone()
        return new

    @property
    def batch_size(self):
        return self.array.shape[0]

    @property
    def array_size(self):
        return self.array.shape[1]

    @property
    def device(self):
        return self.array.device
