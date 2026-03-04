# FDAD: Fully Differentiable Algorithm Discovery
#
# Discovers optimal algorithms from raw input-output data using
# gradient descent over a soft program space built on AlgoVision-style
# differentiable primitives.

from fdad.config import FDADConfig
from fdad.memory import DifferentiableMemory
from fdad.primitives import (
    PrimitiveOp, CompareAndSwap, CopyElement, ShiftRight,
    SoftMinSelect, PointerIncrement, PointerDecrement,
    PointerReset, WriteRegister, NoOp,
)
from fdad.soft_program import SoftProgramStep, SoftProgramBlock, SoftProgram
from fdad.vocabulary import Vocabulary, CompoundOp, UtilityMetrics
from fdad.distribution_router import DistributionClassifier, DistributionAwareProgram
from fdad.discretization import DiscretizationScheduler, extract_discrete_program
from fdad.data import SortingDataGenerator
from fdad.losses import sorting_loss, pairwise_ordering_loss, entropy_loss, sparsity_loss, total_loss
from fdad.trainer import FDADTrainer
