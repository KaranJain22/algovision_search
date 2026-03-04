import torch
import pytest


def test_compare_and_swap_sorts_pair():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import CompareAndSwap

    op = CompareAndSwap()
    # Array [5, 1] should be swapped to [1, 5]
    array = torch.tensor([[5.0, 1.0]])
    mem = DifferentiableMemory(array)

    ptr_i = torch.tensor([1.0, 0.0])  # position 0
    ptr_j = torch.tensor([0.0, 1.0])  # position 1

    result = op(mem, [ptr_i, ptr_j], torch.tensor([]), beta=10.0)
    # After compare-and-swap, position 0 should be smaller
    assert result.array[0, 0].item() < result.array[0, 1].item()


def test_compare_and_swap_no_swap_needed():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import CompareAndSwap

    op = CompareAndSwap()
    array = torch.tensor([[1.0, 5.0]])
    mem = DifferentiableMemory(array)

    ptr_i = torch.tensor([1.0, 0.0])
    ptr_j = torch.tensor([0.0, 1.0])

    result = op(mem, [ptr_i, ptr_j], torch.tensor([]), beta=10.0)
    # Should remain approximately the same
    assert torch.allclose(result.array, array, atol=0.1)


def test_copy_element():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import CopyElement

    op = CopyElement()
    array = torch.tensor([[10.0, 20.0, 30.0]])
    mem = DifferentiableMemory(array)

    ptr_src = torch.tensor([1.0, 0.0, 0.0])  # position 0: value 10
    ptr_dst = torch.tensor([0.0, 0.0, 1.0])  # position 2

    result = op(mem, [ptr_src, ptr_dst], torch.tensor([]), beta=10.0)
    assert torch.allclose(result.array[0, 2], torch.tensor(10.0), atol=1e-5)


def test_noop():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import NoOp

    op = NoOp()
    array = torch.tensor([[1.0, 2.0, 3.0]])
    mem = DifferentiableMemory(array)
    original = mem.array.clone()

    result = op(mem, [], torch.tensor([]), beta=10.0)
    assert torch.allclose(result.array, original)


def test_shift_right():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import ShiftRight

    op = ShiftRight()
    array = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mem = DifferentiableMemory(array)

    result = op(mem, [], torch.tensor([]), beta=10.0)
    expected = torch.tensor([[4.0, 1.0, 2.0, 3.0]])
    assert torch.allclose(result.array, expected)


def test_gradient_flow_through_cas():
    from fdad.memory import DifferentiableMemory
    from fdad.primitives import CompareAndSwap

    op = CompareAndSwap()
    array = torch.randn(4, 8, requires_grad=True)
    mem = DifferentiableMemory(array)

    ptr_i = torch.softmax(torch.randn(8), dim=0)
    ptr_j = torch.softmax(torch.randn(8), dim=0)

    result = op(mem, [ptr_i, ptr_j], torch.tensor([]), beta=10.0)
    loss = result.array.sum()
    loss.backward()
    assert array.grad is not None
    assert not torch.all(array.grad == 0)


def test_get_default_ops():
    from fdad.primitives import get_default_ops
    ops = get_default_ops()
    assert len(ops) == 8
    names = [op.name for op in ops]
    assert 'compare_and_swap' in names
    assert 'noop' in names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
