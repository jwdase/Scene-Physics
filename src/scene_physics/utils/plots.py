import numpy as np
import matplotlib.pyplot as plt


def plot_location_scores(locations, scores, save_path=None, cmap='viridis'):
    """
    Plot sample locations on an x-y scatter plot with color-coded likelihood scores.

    Args:
        locations: List of (x, z) tuples or numpy array of shape (N, 2)
        scores: List or array of likelihood scores corresponding to each location
        save_path: Optional path to save the plot (e.g., 'recordings/mc/samples.png')
        cmap: Matplotlib colormap name for score coloring (default: 'viridis')

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    locations = np.array(locations)
    scores = np.array(scores)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        locations[:, 0],
        locations[:, 1],
        c=scores,
        cmap=cmap,
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Likelihood Score')

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Sample Locations with Likelihood Scores')
    ax.set_aspect('equal', adjustable='box')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
