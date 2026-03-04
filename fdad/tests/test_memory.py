import torch
import pytest


def test_differentiable_memory_creation():
    from fdad.memory import DifferentiableMemory
    array = torch.randn(4, 8)
    mem = DifferentiableMemory(array, num_registers=4)
    assert mem.array.shape == (4, 8)
    assert mem.registers.shape == (4, 4)
    assert mem.batch_size == 4
    assert mem.array_size == 8


def test_soft_read():
    from fdad.memory import DifferentiableMemory
    array = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mem = DifferentiableMemory(array)
    # One-hot pointer at position 2
    ptr = torch.tensor([0.0, 0.0, 1.0, 0.0])
    result = mem.soft_read(ptr)
    assert torch.allclose(result, torch.tensor([3.0]), atol=1e-6)


def test_soft_read_gradient():
    from fdad.memory import DifferentiableMemory
    array = torch.randn(2, 4, requires_grad=True)
    mem = DifferentiableMemory(array)
    ptr = torch.softmax(torch.randn(4), dim=0)
    result = mem.soft_read(ptr)
    result.sum().backward()
    assert array.grad is not None


def test_soft_write():
    from fdad.memory import DifferentiableMemory
    array = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mem = DifferentiableMemory(array)
    # Write 10.0 at position 1 (one-hot)
    ptr = torch.tensor([0.0, 1.0, 0.0, 0.0])
    mem.soft_write(ptr, torch.tensor([10.0]))
    assert torch.allclose(mem.array[0, 1], torch.tensor(10.0), atol=1e-6)
    # Other positions unchanged
    assert torch.allclose(mem.array[0, 0], torch.tensor(1.0), atol=1e-6)


def test_soft_swap():
    from fdad.memory import DifferentiableMemory
    array = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mem = DifferentiableMemory(array)
    ptr_i = torch.tensor([1.0, 0.0, 0.0, 0.0])  # position 0
    ptr_j = torch.tensor([0.0, 0.0, 0.0, 1.0])  # position 3
    mem.soft_swap(ptr_i, ptr_j)
    assert torch.allclose(mem.array[0, 0], torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(mem.array[0, 3], torch.tensor(1.0), atol=1e-5)


def test_soft_compare():
    from fdad.memory import DifferentiableMemory
    array = torch.tensor([[5.0, 1.0, 3.0, 2.0]])
    mem = DifferentiableMemory(array)
    ptr_i = torch.tensor([1.0, 0.0, 0.0, 0.0])  # position 0: value 5
    ptr_j = torch.tensor([0.0, 1.0, 0.0, 0.0])  # position 1: value 1
    p = mem.soft_compare(ptr_i, ptr_j, beta=10.0)
    # 5 > 1, so p should be close to 1
    assert p.item() > 0.9


def test_clone_independence():
    from fdad.memory import DifferentiableMemory
    array = torch.tensor([[1.0, 2.0, 3.0]])
    mem = DifferentiableMemory(array)
    mem_clone = mem.clone()
    mem_clone.array[0, 0] = 99.0
    # Original should be unchanged
    assert mem.array[0, 0].item() == 1.0


def test_merge():
    from fdad.memory import DifferentiableMemory
    a1 = torch.tensor([[1.0, 2.0, 3.0]])
    a2 = torch.tensor([[10.0, 20.0, 30.0]])
    mem1 = DifferentiableMemory(a1)
    mem2 = DifferentiableMemory(a2)
    mem1.merge(mem2, p=torch.tensor([0.5]))
    expected = torch.tensor([[5.5, 11.0, 16.5]])
    assert torch.allclose(mem1.array, expected, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
