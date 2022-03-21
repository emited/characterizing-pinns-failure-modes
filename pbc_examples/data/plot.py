from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.utils.data as data

def plot_solution_2d(U_pred, x, t):
    print('ok')

def plot_solution_1d(U_pred, x, t, ax=None, title=None):
    import matplotlib.pyplot as plt
    """Visualize u_predicted."""

    if ax is None:
        fig, ax = plt.gcf(), plt.gca()

    # colorbar for prediction: set min/max to ground truth solution.
    # U_pred.sum(1, keepdims=True).dot(U_pred.sum(0, keepdims=True))
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto', vmin=U_pred.min().item(), vmax=U_pred.max().item())
    # if X_collocation is not None:
    #     ax.scatter(X_collocation[..., [1]], X_collocation[..., [0]], marker='x', c='black')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xlabel('t', fontweight='bold', size=30)
    ax.set_ylabel('x', fontweight='bold', size=30)
    ax.set_title(title)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    ax.tick_params(labelsize=15)


def plot_latents(px, zt, x, t, labels=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18,6))
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cmaps = [c + '_r' for c in cmaps]
    plt.subplot(1, 2, 1)
    plt.title(f'zt')
    for i, zti in enumerate(zt):
        plt.scatter(zti[:, 0], zti[:, 1], c=t[i],
                label=labels[i],
                cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
    # plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.subplot(1, 2, 2)

    plt.title(f'px')
    for i, pxi in enumerate(px):
        plt.scatter(pxi[:, 0], pxi[:, 1], c=x[i],
                label=labels[i],
                cmap=cmaps[i], alpha=1)  # yellow: 1, blue:  0
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    plt.tight_layout()


