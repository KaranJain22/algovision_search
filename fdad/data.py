import torch
from typing import Tuple, Optional


class SortingDataGenerator:
    """Generates sorting input-output pairs with varied distributions.

    Supports 6 distribution types for discovering distribution-specific
    optimal sorting algorithms.
    """

    DISTRIBUTIONS = [
        'uniform',          # Uniform random [0, 1]
        'gaussian',         # Normal distribution
        'nearly_sorted',    # Sorted + small perturbations
        'reverse_sorted',   # Reverse sorted + small perturbations
        'few_unique',       # Few unique values repeated
        'bimodal',          # Two clusters of values
    ]

    def generate_batch(self, batch_size, array_size, distribution='uniform',
                       device='cpu'):
        """Generate sorting input-output pairs.

        Args:
            batch_size: number of samples
            array_size: length of each array
            distribution: one of DISTRIBUTIONS
            device: torch device
        Returns:
            arrays: (B, N) unsorted input arrays
            targets: (B, N) ground truth sorted arrays
            dist_label: int distribution index
        """
        B, N = batch_size, array_size
        dist_idx = (self.DISTRIBUTIONS.index(distribution)
                    if distribution in self.DISTRIBUTIONS else 0)

        if distribution == 'uniform':
            arrays = torch.rand(B, N, device=device)

        elif distribution == 'gaussian':
            arrays = torch.randn(B, N, device=device)

        elif distribution == 'nearly_sorted':
            base = torch.arange(N, dtype=torch.float32, device=device)
            arrays = base.unsqueeze(0).expand(B, -1).clone()
            arrays += torch.randn(B, N, device=device) * 0.3

        elif distribution == 'reverse_sorted':
            base = torch.arange(N - 1, -1, -1, dtype=torch.float32,
                                device=device)
            arrays = base.unsqueeze(0).expand(B, -1).clone()
            arrays += torch.randn(B, N, device=device) * 0.1

        elif distribution == 'few_unique':
            num_unique = max(2, N // 4)
            values = torch.rand(B, num_unique, device=device)
            indices = torch.randint(0, num_unique, (B, N), device=device)
            arrays = values.gather(1, indices)

        elif distribution == 'bimodal':
            half = N // 2
            low = torch.rand(B, half, device=device) * 0.3
            high = torch.rand(B, N - half, device=device) * 0.3 + 0.7
            arrays = torch.cat([low, high], dim=-1)
            # Shuffle within each sample
            perm = torch.argsort(torch.rand(B, N, device=device), dim=-1)
            arrays = arrays.gather(1, perm)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        targets = arrays.sort(dim=-1).values
        return arrays, targets, dist_idx

    def generate_mixed_batch(self, batch_size, array_size, device='cpu'):
        """Generate batch with mixed distributions (for router training).

        Each distribution gets an equal share of the batch.

        Args:
            batch_size: total number of samples
            array_size: length of each array
            device: torch device
        Returns:
            arrays: (B, N) unsorted
            targets: (B, N) sorted
            labels: (B,) int distribution indices
        """
        num_dists = len(self.DISTRIBUTIONS)
        per_dist = batch_size // num_dists
        all_arrays, all_targets, all_labels = [], [], []

        for i, dist in enumerate(self.DISTRIBUTIONS):
            n = per_dist if i < num_dists - 1 else batch_size - per_dist * i
            if n <= 0:
                continue
            a, t, _ = self.generate_batch(n, array_size, dist, device)
            all_arrays.append(a)
            all_targets.append(t)
            all_labels.extend([i] * n)

        arrays = torch.cat(all_arrays, dim=0)
        targets = torch.cat(all_targets, dim=0)
        labels = torch.tensor(all_labels, device=device)

        # Shuffle
        perm = torch.randperm(arrays.shape[0], device=device)
        return arrays[perm], targets[perm], labels[perm]

    def generate_curriculum_batch(self, batch_size, min_size, max_size,
                                  step, warmup_steps=5000, device='cpu'):
        """Generate batch with curriculum learning on array size.

        Starts with small arrays and gradually increases to max_size.

        Args:
            batch_size: number of samples
            min_size: minimum array size
            max_size: maximum array size
            step: current training step
            warmup_steps: steps over which to ramp up size
            device: torch device
        Returns:
            arrays, targets, labels (same as generate_mixed_batch)
        """
        progress = min(step / max(warmup_steps, 1), 1.0)
        current_max = int(min_size + (max_size - min_size) * progress)
        current_max = max(current_max, min_size)

        # Random size within current range
        array_size = torch.randint(min_size, current_max + 1, (1,)).item()
        return self.generate_mixed_batch(batch_size, array_size, device)
