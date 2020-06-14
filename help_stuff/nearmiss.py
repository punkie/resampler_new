import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import TomekLinks

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
    ax.legend()
    plt.show()

def tomek_links():
    # minority class
    X_minority = np.transpose([[1.4, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [0.4, 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])
    # majority class
    X_majority = np.transpose([[2.1, 1.5, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5],
                               [1.5, 2.2, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3]])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.scatter(X_majority[:, 0], X_majority[:, 1],
    #            label='Negative class', s=200, marker='_')
    #
    # ax.scatter(X_minority[:, 0], X_minority[:, 1],
    #            label='Positive class', s=200, marker='+')
    #
    # # highlight the samples of interest
    # ax.scatter([X_minority[-1, 0], X_majority[1, 0]],
    #            [X_minority[-1, 1], X_majority[1, 1]],
    #            label='Tomek link', s=200, alpha=0.3)
    # ax.set_title('Illustration of a Tomek link')
    # make_plot_despine(ax)
    # fig.tight_layout()

    sampler = TomekLinks()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax_arr = (ax1, ax2)
    title_arr = ('Removing only majority samples',
                 'Removing all samples')
    for ax, title, sampler in zip(ax_arr,
                                  title_arr,
                                  [TomekLinks(sampling_strategy='auto'),
                                   TomekLinks(sampling_strategy='all')]):
        X_res, y_res = sampler.fit_resample(np.vstack((X_majority, X_minority)),
                                            np.array([0] * X_majority.shape[0] +
                                                     [1] * X_minority.shape[0]))
        ax.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1],
                   label='Minority class', s=200, marker='+')
        ax.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1],
                   label='Majority class', s=200, marker='_')

        # highlight the samples of interest
        ax.scatter([X_minority[-1, 0], X_majority[1, 0]],
                   [X_minority[-1, 1], X_majority[1, 1]],
                   label='Tomek link', s=200, alpha=0.3)

        ax.set_title(title)
        make_plot_despine(ax)
    fig.tight_layout()

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

    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Средно разстояние={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    dist, ind = nearest_neighbors.kneighbors(X_majority)
    av_dist_index = np.argsort([sum(d)/3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[:7]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателните примери избрани от NM1', s=200, alpha=0.3, color='g')

    ax.set_title('NearMiss-1')
    make_plot_despine(ax)


def nm2():
    # minority class
    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])
    # majority class
    X_majority = np.transpose([[2.1, 2.13, 2.12, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5, 1.5],
                               [1.5, 2.7, 2.1, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3, 2.2]])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателен клас', s=200, marker='_')
    ax.scatter(X_minority[:, 0], X_minority[:, 1],
               label='Положителен клас', s=200, marker='+')

    nearest_neighbors = NearestNeighbors(n_neighbors=X_minority.shape[0])
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist = dist[:, -3::]
    ind = ind[:, -3::]
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Средно разстояние={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    ax.set_title('NearMiss-2')

    dist, ind = nearest_neighbors.kneighbors(X_majority)
    dist = dist[:, -3::]
    av_dist_index = np.argsort([sum(d) / 3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[:7]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателните примери избрани от NM2', s=200, alpha=0.3, color='g')

    make_plot_despine(ax)


def nm3():
    # minority class
    X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.8, 0.6, 0.55],
                               [1., 1.5, 1.7, 2.5, 2.0, 1.2, 0.55]])
    # majority class
    X_majority = np.transpose([[2.1, 1.5, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45, 3.00, 3.1, 1.5],
                               [1.5, 2.2, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9, 1.00, 2.0, 0.3]])
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
               label='М най-близки съседи на + примери', s=800, alpha=0.5, color='y')
    nearest_neighbors = NearestNeighbors(n_neighbors=3)
    nearest_neighbors.fit(X_minority)
    dist, ind = nearest_neighbors.kneighbors(X_majority[:2, :])
    dist_avg = dist.sum(axis=1) / 3

    for positive_idx, (neighbors, distance, color) in enumerate(
            zip(ind, dist_avg, ['g', 'r'])):
        for make_plot, sample_idx in enumerate(neighbors):
            ax.plot([X_majority[positive_idx, 0], X_minority[sample_idx, 0]],
                    [X_majority[positive_idx, 1], X_minority[sample_idx, 1]],
                    '--' + color, alpha=0.3,
                    label='Средно разстояние={:.2f}'.format(distance)
                    if make_plot == 0 else "")
    ax.set_title('NearMiss-3')
    dist, ind = nearest_neighbors.kneighbors(X_majority)
    av_dist_index = np.argsort([sum(d) / 3 for d in dist])
    X_maj_subset = np.asarray([X_majority[idx] for idx in av_dist_index[-7:]])
    X_majority = X_maj_subset
    ax.scatter(X_majority[:, 0], X_majority[:, 1],
               label='Отрицателните примери избрани от NM3', s=200, alpha=0.3, color='g')
    make_plot_despine(ax)

    fig.tight_layout()


def nm4():

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
    ax.set_title('NearMiss-4')
    make_plot_despine(ax)



if __name__ == "__main__":
    # nm1()
    # nm2()
    # nm3()
    # nm4()
    tomek_links()