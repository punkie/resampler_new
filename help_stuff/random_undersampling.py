import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import NearestNeighbors

print(__doc__)

rng = np.random.RandomState(18)


def make_plot_despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([0., 3.5])
    ax.set_ylim([0., 3.5])
    ax.set_xlabel(r'$X_1$')
    ax.set_ylabel(r'$X_2$')
    ax.legend(loc='upper left')
    plt.show()


def nm1():

    # minority class
    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])
    # majority class
    X_majority = np.transpose([[2.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5, 1.5],
                               [1.5, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3, 2.2]])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателен клас', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Положителен клас', s=200, marker='+')

    selected_idx = take_unique_indexes(9, 7)

    X_majority = X_majority[np.unique(selected_idx), :]
    # nearest_neighbors = NearestNeighbors(n_neighbors=3)
    # nearest_neighbors.fit(X_minority)
    # dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    # dist_avg = dist.sum(axis=1) / 3

    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Произволно избрани отрицателни примери', s=200, alpha=0.3, color='g')

    # for positive_idx, (neighbors, distance, color) in enumerate(
    #         zip(ind, dist_avg, ['g', 'r'])):
    #     for make_plot, sample_idx in enumerate(neighbors):
    #         ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
    #                 [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
    #                 '--' + color, alpha=0.3,
    #                 label='Средно разстояние={:.2f}'.format(distance)
    #                 if make_plot == 0 else "")
    ax.set_title('Random Under-sampling')
    make_plot_despine(ax)
    fig.tight_layout()

def take_unique_indexes(high_index, amount):
    indexes = np.unique([])
    while len(indexes) != amount:
        indexes = np.unique(np.random.randint(0, high_index, amount))
    return indexes


def nm3():
    # minority class
    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])
    # majority class
    X_majority = np.transpose([[2.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45],
                               [1.5, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9]])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателен клас', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Положителен клас', s=200, marker='+')

    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_majority)

    # select only the majority point of interest
    selected_idx = nearest_neighbors.kneighbors(X_minority, return_distance=False)
    X_majority = X_majority[np.unique(selected_idx), :]
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='М най-близки съседи', s=200, alpha=0.3, color='g')
    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['r', 'g'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Средно разстояние={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    ax.set_title('NearMiss-3')
    make_plot_despine(ax)

    fig.tight_layout()

if __name__ == "__main__":
    nm1()