import torch
import pytest


def test_all_distributions():
    from fdad.data import SortingDataGenerator
    gen = SortingDataGenerator()

    for dist in gen.DISTRIBUTIONS:
        arrays, targets, label = gen.generate_batch(16, 8, dist)
        assert arrays.shape == (16, 8), f"Failed for {dist}"
        assert targets.shape == (16, 8), f"Failed for {dist}"
        # Verify targets are actually sorted
        assert torch.all(targets[:, :-1] <= targets[:, 1:]), \
            f"Targets not sorted for {dist}"


def test_mixed_batch():
    from fdad.data import SortingDataGenerator
    gen = SortingDataGenerator()

    arrays, targets, labels = gen.generate_mixed_batch(24, 8)
    assert arrays.shape == (24, 8)
    assert targets.shape == (24, 8)
    assert labels.shape == (24,)
    # All targets should be sorted
    assert torch.all(targets[:, :-1] <= targets[:, 1:])


def test_content_preservation():
    """Verify that sorted target has same values as input."""
    from fdad.data import SortingDataGenerator
    gen = SortingDataGenerator()

    for dist in ['uniform', 'gaussian', 'bimodal']:
        arrays, targets, _ = gen.generate_batch(8, 6, dist)
        # Sort both and compare
        arrays_sorted = arrays.sort(dim=-1).values
        assert torch.allclose(arrays_sorted, targets, atol=1e-6), \
            f"Content mismatch for {dist}"


def test_different_sizes():
    from fdad.data import SortingDataGenerator
    gen = SortingDataGenerator()

    for size in [4, 8, 12, 16, 24, 32]:
        arrays, targets, _ = gen.generate_batch(8, size, 'uniform')
        assert arrays.shape == (8, size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
