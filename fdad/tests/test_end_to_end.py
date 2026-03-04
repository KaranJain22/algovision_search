import torch
import pytest


def test_single_program_forward_backward():
    """Test that a single SoftProgram can do forward + backward."""
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary
    from fdad.losses import total_loss

    torch.manual_seed(42)
    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=8,
    )

    array = torch.randn(4, 8)
    target = array.sort(dim=-1).values

    output, _ = prog(array, vocab, tau=1.0, beta=10.0)
    loss = total_loss(output, target, prog, entropy_weight=0.01)

    loss.backward()
    assert loss.item() > 0


def test_distribution_aware_forward_backward():
    """Test the full DistributionAwareProgram pipeline."""
    from fdad.distribution_router import DistributionAwareProgram
    from fdad.vocabulary import Vocabulary
    from fdad.losses import total_loss

    torch.manual_seed(42)
    vocab = Vocabulary()
    model = DistributionAwareProgram(
        num_programs=2,
        num_outer_iters=2,
        num_inner_steps=2,
        vocab=vocab,
        max_array_size=8,
    )

    array = torch.randn(4, 8)
    target = array.sort(dim=-1).values

    output, route_weights, step_infos = model(array, tau=1.0, beta=10.0)
    assert output.shape == (4, 8)
    assert route_weights.shape == (4, 2)

    loss = total_loss(output, target, model, entropy_weight=0.01)
    loss.backward()
    assert loss.item() > 0


def test_training_reduces_loss():
    """Verify that a few training steps reduce the loss."""
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary
    from fdad.losses import total_loss
    from fdad.data import SortingDataGenerator

    torch.manual_seed(42)
    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=4,
    )
    optimizer = torch.optim.Adam(prog.parameters(), lr=1e-2)
    data_gen = SortingDataGenerator()

    initial_loss = None
    final_loss = None

    for step in range(50):
        arrays, targets, _ = data_gen.generate_batch(32, 4, 'uniform')

        output, _ = prog(arrays, vocab, tau=1.0, beta=10.0)
        loss = total_loss(output, targets, prog, entropy_weight=0.0)

        if step == 0:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prog.parameters(), 1.0)
        optimizer.step()

        final_loss = loss.item()

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


def test_discretization_scheduler():
    from fdad.discretization import DiscretizationScheduler

    sched = DiscretizationScheduler(
        total_steps=1000, exploration_frac=0.3, refinement_frac=0.4
    )

    # Phase boundaries
    assert sched.get_phase(0) == 'exploration'
    assert sched.get_phase(299) == 'exploration'
    assert sched.get_phase(300) == 'refinement'
    assert sched.get_phase(699) == 'refinement'
    assert sched.get_phase(700) == 'collapse'
    assert sched.get_phase(999) == 'collapse'

    # Temperature should decrease monotonically
    taus = [sched.get_tau(s) for s in range(1000)]
    for i in range(len(taus) - 1):
        assert taus[i] >= taus[i+1] - 1e-6, \
            f"Tau increased at step {i}: {taus[i]} -> {taus[i+1]}"

    # Final tau should be close to 0.01
    assert taus[-1] < 0.02


def test_extract_discrete_program():
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary
    from fdad.discretization import extract_discrete_program

    torch.manual_seed(42)
    vocab = Vocabulary()
    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=8,
    )

    extracted = extract_discrete_program(prog, vocab)
    assert 'loop_count' in extracted
    assert 'steps' in extracted
    assert 'description' in extracted
    assert isinstance(extracted['description'], str)


def test_vocabulary_expansion():
    """Test that vocabulary expansion works correctly."""
    from fdad.soft_program import SoftProgram
    from fdad.vocabulary import Vocabulary

    torch.manual_seed(42)
    vocab = Vocabulary()
    initial_size = vocab.size

    prog = SoftProgram(
        num_outer_iters=4,
        num_inner_steps=4,
        vocab_size=vocab.size,
        max_array_size=8,
    )

    # Manually set dominant ops to trigger expansion
    with torch.no_grad():
        for step in prog.inner_block.steps:
            step.op_logits.fill_(-10.0)
            step.op_logits[0] = 10.0  # CompareAndSwap dominant

    candidates = vocab.check_expansion(prog, threshold=0.8)
    if candidates:
        start, end, name = candidates[0]
        vocab.expand(prog, start, end, name)
        assert vocab.size == initial_size + 1

        # Verify logits were expanded
        for step in prog.inner_block.steps:
            assert step.op_logits.shape[0] == initial_size + 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
