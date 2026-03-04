from dataclasses import dataclass, field
from typing import List


@dataclass
class FDADConfig:
    """Configuration for the FDAD framework."""

    # Array sizes
    min_array_size: int = 4
    max_array_size: int = 16    # train up to 16, test OOD up to 32

    # Program structure (weight-tied architecture)
    num_outer_iters: int = 16   # K: outer loop count (~max_array_size)
    num_inner_steps: int = 8    # T: instructions per shared inner block
    num_programs: int = 4       # specialized programs (one per distribution cluster)
    num_registers: int = 4      # auxiliary scalar registers
    max_ptrs: int = 2           # max pointer args per operation
    max_scalars: int = 2        # max scalar args per operation

    # Training
    total_steps: int = 50_000
    batch_size: int = 64
    lr: float = 1e-3
    beta: float = 10.0          # AlgoVision inverse temperature
    grad_clip: float = 1.0

    # Vocabulary expansion
    expansion_check_freq: int = 1000
    expansion_threshold: float = 0.8    # min dominant op weight to consider
    min_subseq_length: int = 2
    max_subseq_length: int = 5
    expansion_grad_multiplier: float = 2.0  # gradient must exceed mean by this factor
    co_occurrence_window: int = 500         # steps of history to analyze

    # Evaluation
    eval_freq: int = 500
    checkpoint_freq: int = 5000
    eval_batch_size: int = 256
    ood_sizes: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 24, 32])

    # Discretization schedule
    exploration_frac: float = 0.3   # Phase 1: high tau, explore
    refinement_frac: float = 0.4    # Phase 2: medium tau, vocab expansion
    # remaining 0.3 = Phase 3: anneal tau -> 0.01, collapse

    # Device
    device: str = 'cuda'

    # Logging
    log_freq: int = 100
    checkpoint_dir: str = './fdad_checkpoints'

    @property
    def exploration_end(self) -> int:
        return int(self.total_steps * self.exploration_frac)

    @property
    def refinement_end(self) -> int:
        return int(self.total_steps * (self.exploration_frac + self.refinement_frac))
