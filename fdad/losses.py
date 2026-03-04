import torch
import torch.nn.functional as F


def sorting_loss(predicted, target):
    """Primary loss: MSE between predicted and ground truth sorted array.

    Args:
        predicted: (B, N) predicted sorted array
        target: (B, N) ground truth sorted array
    Returns:
        scalar loss
    """
    return F.mse_loss(predicted, target)


def pairwise_ordering_loss(predicted, beta=10.0):
    """Auxiliary loss: encourage all adjacent pairs to be in sorted order.

    Uses sigmoid relaxation (AlgoVision GT pattern) for differentiability.
    L = mean_i sigmoid((pred[i] - pred[i+1]) * beta)

    Args:
        predicted: (B, N) predicted output
        beta: inverse temperature for sigmoid
    Returns:
        scalar loss
    """
    diffs = predicted[:, :-1] - predicted[:, 1:]
    violations = torch.sigmoid(diffs * beta)
    return violations.mean()


def content_preservation_loss(predicted, target):
    """Ensure output is a permutation of input (same multiset of values).

    Checks that sorted(predicted) == sorted(target).
    Since target is already sorted, compare sorted(predicted) to target.

    Args:
        predicted: (B, N) predicted output
        target: (B, N) ground truth sorted array
    Returns:
        scalar loss
    """
    predicted_sorted = torch.sort(predicted, dim=-1).values
    return F.mse_loss(predicted_sorted, target)


def entropy_loss(program):
    """Compute entropy of operation selection distributions.

    During discretization, minimizing this encourages peaked
    (near-deterministic) selections.

    Handles both SoftProgram (with inner_block) and bare SoftProgramBlock.

    Args:
        program: SoftProgram or SoftProgramBlock
    Returns:
        scalar average entropy across all steps
    """
    steps = _get_steps(program)
    if not steps:
        return torch.tensor(0.0)

    total_entropy = torch.tensor(0.0, device=steps[0].op_logits.device)
    for step in steps:
        probs = F.softmax(step.op_logits, dim=-1)
        step_entropy = -(probs * (probs + 1e-8).log()).sum()
        total_entropy = total_entropy + step_entropy
    return total_entropy / len(steps)


def sparsity_loss(program, noop_idx=-1):
    """Encourage the program to use fewer active steps (more NoOps).

    L1 penalty on non-NoOp selection probabilities.

    Args:
        program: SoftProgram or SoftProgramBlock
        noop_idx: index of NoOp in the vocabulary (default: last)
    Returns:
        scalar loss
    """
    steps = _get_steps(program)
    if not steps:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=steps[0].op_logits.device)
    for step in steps:
        probs = F.softmax(step.op_logits, dim=-1)
        active_prob = 1.0 - probs[noop_idx]
        total = total + active_prob
    return total / len(steps)


def total_loss(predicted, target, program, entropy_weight,
               sparsity_weight=0.01, ordering_weight=0.1,
               content_weight=0.0, beta=10.0):
    """Combined loss function.

    L = L_sort + lambda_order * L_order + lambda_entropy * L_entropy
        + lambda_sparse * L_sparse + lambda_content * L_content

    Args:
        predicted: (B, N) predicted output
        target: (B, N) ground truth sorted array
        program: SoftProgram or DistributionAwareProgram
        entropy_weight: weight for entropy regularization
        sparsity_weight: weight for sparsity penalty
        ordering_weight: weight for pairwise ordering loss
        content_weight: weight for content preservation loss
        beta: inverse temperature
    Returns:
        scalar total loss
    """
    l_sort = sorting_loss(predicted, target)
    l_order = pairwise_ordering_loss(predicted, beta)

    # Handle DistributionAwareProgram by iterating over its sub-programs
    programs = _get_programs(program)
    l_entropy = sum(entropy_loss(p) for p in programs) / max(len(programs), 1)
    l_sparse = sum(sparsity_loss(p) for p in programs) / max(len(programs), 1)

    loss = l_sort + ordering_weight * l_order

    if entropy_weight > 0:
        loss = loss + entropy_weight * l_entropy
    if sparsity_weight > 0:
        loss = loss + sparsity_weight * l_sparse
    if content_weight > 0:
        l_content = content_preservation_loss(predicted, target)
        loss = loss + content_weight * l_content

    return loss


def _get_steps(program):
    """Extract SoftProgramStep list from various program types."""
    if hasattr(program, 'inner_block'):
        return list(program.inner_block.steps)
    elif hasattr(program, 'steps'):
        return list(program.steps)
    return []


def _get_programs(model):
    """Extract list of SoftProgram instances from model."""
    if hasattr(model, 'programs'):
        return list(model.programs)
    return [model]
