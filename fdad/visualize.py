import torch
import torch.nn.functional as F


def plot_program_heatmap(program, vocabulary, save_path=None):
    """Plot operation selection weights as a heatmap (steps x vocabulary).

    Shows which operation each inner block step prefers.
    After discretization, this should show near-one-hot rows.

    Args:
        program: SoftProgram instance
        vocabulary: Vocabulary instance
        save_path: optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    V = vocabulary.size
    steps = list(program.inner_block.steps)
    T = len(steps)

    heatmap = torch.zeros(T, V)
    for t, step in enumerate(steps):
        probs = F.softmax(step.op_logits[:V], dim=-1).detach()
        heatmap[t, :len(probs)] = probs

    fig, ax = plt.subplots(figsize=(max(8, V * 0.8), max(4, T * 0.4)))
    im = ax.imshow(heatmap.numpy(), aspect='auto', cmap='Blues',
                   vmin=0, vmax=1)
    ax.set_xlabel('Operation')
    ax.set_ylabel('Step')
    ax.set_xticks(range(V))
    ax.set_xticklabels(vocabulary.op_names, rotation=45, ha='right',
                       fontsize=8)
    ax.set_yticks(range(T))
    ax.set_title('Operation Selection Weights')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pointer_trajectories(program, array_size, save_path=None):
    """Plot pointer position distributions across inner block steps.

    Shows where the program is "looking" at each step.

    Args:
        program: SoftProgram instance
        array_size: actual array size (N)
        save_path: optional path to save
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    steps = list(program.inner_block.steps)
    T = len(steps)
    N = array_size

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, T * 0.4)))

    for ptr_idx in range(min(2, steps[0].max_ptrs)):
        heatmap = torch.zeros(T, N)
        for t, step in enumerate(steps):
            probs = F.softmax(step.pointer_logits[ptr_idx, :N],
                              dim=-1).detach()
            heatmap[t] = probs

        ax = axes[ptr_idx]
        im = ax.imshow(heatmap.numpy(), aspect='auto', cmap='Oranges',
                       vmin=0, vmax=1)
        ax.set_xlabel('Array Position')
        ax.set_ylabel('Step')
        ax.set_title(f'Pointer {ptr_idx} Attention')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(history, save_path=None):
    """Plot training metrics over time.

    Args:
        history: list of dicts with keys like 'loss', 'tau', 'phase', etc.
        save_path: optional path to save
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    steps = range(len(history))
    losses = [h.get('loss', 0) for h in history]
    taus = [h.get('tau', 1.0) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(steps, losses, alpha=0.7)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].set_yscale('log')

    axes[1].plot(steps, taus, color='red', alpha=0.7)
    axes[1].set_ylabel('Temperature (tau)')
    axes[1].set_xlabel('Step')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_routing_distribution(route_weights, dist_labels, save_path=None):
    """Show which distributions get routed to which programs.

    Args:
        route_weights: (B, K) routing probabilities
        dist_labels: (B,) int distribution indices
        save_path: optional path to save
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for visualization")
        return

    from fdad.data import SortingDataGenerator

    K = route_weights.shape[1]
    dist_names = SortingDataGenerator.DISTRIBUTIONS

    # Average routing per distribution type
    num_dists = len(dist_names)
    avg_routing = torch.zeros(num_dists, K)
    for d in range(num_dists):
        mask = dist_labels == d
        if mask.any():
            avg_routing[d] = route_weights[mask].mean(dim=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(avg_routing.numpy(), aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1)
    ax.set_xlabel('Program Index')
    ax.set_ylabel('Distribution Type')
    ax.set_xticks(range(K))
    ax.set_yticks(range(num_dists))
    ax.set_yticklabels(dist_names)
    ax.set_title('Average Routing per Distribution')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_discrete_program(extracted, title="Discovered Algorithm"):
    """Pretty-print an extracted discrete program.

    Args:
        extracted: dict from extract_discrete_program()
        title: display title
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(extracted['description'])
    print(f"\nHalt probabilities: {[f'{p:.2f}' for p in extracted['halt_probs']]}")
    print(f"{'='*60}\n")
