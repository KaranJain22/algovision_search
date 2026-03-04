import torch
import torch.nn as nn
import torch.nn.functional as F
from fdad.soft_program import SoftProgram


class DistributionClassifier(nn.Module):
    """Classifies input arrays into distribution categories.

    Computes hand-crafted features characterizing the input distribution:
    - sortedness, reverse-sortedness
    - inversion count (normalized)
    - value range, std dev, mean
    - duplicate ratio
    - log array size

    These features are passed through a small MLP to produce routing logits.
    """

    def __init__(self, num_distributions, feature_dim=16):
        super().__init__()
        self.num_distributions = num_distributions
        self.feature_net = nn.Sequential(
            nn.Linear(8, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_distributions),
        )

    def compute_features(self, array):
        """Extract distribution-characterizing features.

        Args:
            array: (B, N) input array
        Returns:
            (B, 8) feature tensor
        """
        B, N = array.shape
        device = array.device

        # 1. Sortedness: fraction of adjacent pairs in order
        sorted_frac = (array[:, :-1] <= array[:, 1:]).float().mean(
            dim=-1, keepdim=True
        )

        # 2. Reverse-sortedness
        rev_sorted_frac = (array[:, :-1] >= array[:, 1:]).float().mean(
            dim=-1, keepdim=True
        )

        # 3. Normalized inversion count (differentiable proxy)
        # Use pairwise comparison matrix upper triangle
        pair_count = N * (N - 1) / 2
        if pair_count > 0:
            diff_matrix = array.unsqueeze(-1) - array.unsqueeze(-2)  # (B, N, N)
            mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)
            inv_count = (torch.sigmoid(diff_matrix * 5.0) * mask).sum(
                dim=(-1, -2)
            ) / pair_count
        else:
            inv_count = torch.zeros(B, device=device)
        inv_count = inv_count.unsqueeze(-1)

        # 4. Value range (normalized by N)
        val_range = (
            (array.max(dim=-1).values - array.min(dim=-1).values) / max(N, 1)
        ).unsqueeze(-1)

        # 5. Standard deviation
        val_std = array.std(dim=-1, keepdim=True)

        # 6. Duplicate ratio (approximate: fraction of near-equal pairs)
        if pair_count > 0:
            abs_diffs = diff_matrix.abs()
            # Use soft threshold for differentiability
            dup_ratio = (
                torch.sigmoid(-(abs_diffs - 0.01) * 100) * mask
            ).sum(dim=(-1, -2)) / pair_count
        else:
            dup_ratio = torch.zeros(B, device=device)
        dup_ratio = dup_ratio.unsqueeze(-1)

        # 7. Log array size
        log_size = torch.full(
            (B, 1), torch.log(torch.tensor(float(N))).item(),
            device=device
        )

        # 8. Mean value
        val_mean = array.mean(dim=-1, keepdim=True)

        return torch.cat([
            sorted_frac, rev_sorted_frac, inv_count, val_range,
            val_std, dup_ratio, log_size, val_mean,
        ], dim=-1)

    def forward(self, array):
        """Compute soft routing weights.

        Args:
            array: (B, N) input array
        Returns:
            (B, num_distributions) softmax routing weights
        """
        features = self.compute_features(array)
        logits = self.feature_net(features)
        return logits  # raw logits; softmax applied in router


class DistributionAwareProgram(nn.Module):
    """Top-level module: routes inputs to specialized soft programs.

    Maintains K soft programs (one per distribution type) and a classifier
    that soft-routes inputs. During discretization, routing becomes hard.
    """

    def __init__(self, num_programs, num_outer_iters, num_inner_steps,
                 vocab, max_array_size, num_registers=4,
                 max_ptrs=2, max_scalars=2):
        super().__init__()
        self.classifier = DistributionClassifier(num_programs)
        self.programs = nn.ModuleList([
            SoftProgram(
                num_outer_iters, num_inner_steps, vocab.size,
                max_array_size, num_registers, max_ptrs, max_scalars,
            )
            for _ in range(num_programs)
        ])
        self.vocab = vocab
        self.num_programs = num_programs

    def forward(self, input_array, tau, beta):
        """Forward pass with soft routing.

        1. Classify distribution -> routing weights (B, K)
        2. Run each program -> K outputs (B, N)
        3. Weighted combination of outputs

        Args:
            input_array: (B, N) unsorted array
            tau: Gumbel-Softmax temperature
            beta: inverse temperature
        Returns:
            final_output: (B, N) sorted result
            route_weights: (B, K) routing probabilities
            all_step_info: list of step_info dicts per program
        """
        route_logits = self.classifier(input_array)
        route_weights = F.softmax(route_logits / max(tau, 1e-6), dim=-1)

        outputs = []
        all_step_info = []
        for prog in self.programs:
            out, info = prog(input_array, self.vocab, tau, beta)
            outputs.append(out)
            all_step_info.append(info)

        # Weighted combination: (B, K, N) * (B, K, 1) -> sum -> (B, N)
        stacked = torch.stack(outputs, dim=1)  # (B, K, N)
        weights_expanded = route_weights.unsqueeze(-1)  # (B, K, 1)
        final_output = (stacked * weights_expanded).sum(dim=1)  # (B, N)

        return final_output, route_weights, all_step_info
