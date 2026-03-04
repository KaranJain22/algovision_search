import torch
import pytest


def test_soft_program_step_forward():
    from fdad.soft_program import SoftProgramStep
    from fdad.vocabulary import Vocabulary
    from fdad.memory import DifferentiableMemory

    vocab = Vocabulary()
    step = SoftProgramStep(vocab.size, max_array_size=8)
    array = torch.randn(4, 8)
    mem = DifferentiableMemory(array)

    result = step(mem, vocab, tau=1.0, beta=10.0, array_size=8)
    assert result.array.shape == (4, 8)
    assert result.registers.shape == (4, 4)


def test_soft_program_block_forward():
    from fdad.soft_program import SoftProgramBlock
    from fdad.vocabulary import Vocabulary
    from fdad.memory import DifferentiableMemory

    vocab = Vocabulary()
    block = SoftProgramBlock(num_steps=4, vocab_size=vocab.size,
                             max_array_size=8)
    array = torch.randn(2, 8)
    mem = DifferentiableMemory(array)

    result = block(mem, vocab, tau=1.0, beta=10.0, array_size=8)
    assert result.array.shape == (2, 8)


def test_soft_program_forward():
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary

    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=8,
    )

    array = torch.randn(2, 8)
    output, step_info = prog(array, vocab, tau=1.0, beta=10.0)

    assert output.shape == (2, 8)
    assert 'op_weights' in step_info
    assert 'halt_probs' in step_info
    assert len(step_info['op_weights']) == 4 * 4  # outer_iters * inner_steps


def test_soft_program_gradient_flow():
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary

    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=2,
        num_inner_steps=2,
        vocab_size=vocab.size,
        max_array_size=4,
    )

    array = torch.randn(2, 4)
    target = array.sort(dim=-1).values

    output, _ = prog(array, vocab, tau=1.0, beta=10.0)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    # Check that gradients flow to program parameters
    has_grad = False
    for p in prog.parameters():
        if p.grad is not None and not torch.all(p.grad == 0):
            has_grad = True
            break
    assert has_grad, "No gradients flowed to program parameters"


def test_soft_program_variable_array_size():
    """Test that program works with different array sizes."""
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary

    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=16,
    )

    for N in [4, 8, 12, 16]:
        array = torch.randn(2, N)
        output, _ = prog(array, vocab, tau=1.0, beta=10.0)
        assert output.shape == (2, N), f"Failed for N={N}"


def test_effective_iters():
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary

    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=8,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=8,
    )

    # Default: halt logits are 0 -> sigmoid(0)=0.5 (not >0.5)
    # So all iterations run
    assert prog.get_effective_iters() == 8

    # Set first halt logit high -> should halt at iter 1
    with torch.no_grad():
        prog.halt_logits[0] = 10.0  # sigmoid(10) ≈ 1.0
    assert prog.get_effective_iters() == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
