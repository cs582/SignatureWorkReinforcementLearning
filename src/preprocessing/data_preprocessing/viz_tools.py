import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, OPTICS


def plot_tokens_history(data, title, columns):
    plt.figure(figsize=(15, 8))
    plt.title(title)

    if columns is not None:
        plt.plot(data[columns].values)
    else:
        plt.plot(data.values, color='b', alpha=0.5)

    step = len(data.index)//6
    ticks = np.arange(0, len(data.index))[::step]
    axis_labels = data.index[::step]

    plt.xticks(ticks=ticks, labels=axis_labels)
    plt.show()


def plot_scatter_2d(data):
    plt.figure(figsize=(15, 8))
    plt.title("t-SNE visualization")
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def plot_components(data):
    plt.figure(figsize=(15, 8))
    plt.title("Showing first 2 components")
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def plot_pca_variance(pca, var_th):
    plt.figure(figsize=(15, 8))

    xi = np.arange(0, len(pca.explained_variance_ratio_))
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, len(pca.explained_variance_ratio_)))
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=var_th, color='r', linestyle='-')
    plt.text(0.5, 0.85, f'{int(var_th * 100)}% cut-off threshold', color='red', fontsize=16)

    plt.show()


def dim_reduction_pca(df, var_th, show_variance):
    if show_variance:
        plot_pca_variance(pca=PCA().fit(df), var_th=var_th)

    pca = PCA(n_components=var_th).fit_transform(df)

    return pca


def train_unsupervised_algo(data, algo_name, min_samples):
    algo = None
    if algo_name == "OPTICS":
        algo = OPTICS(min_samples=min_samples).fit(data)
    if algo_name == "DBSCAN":
        algo = DBSCAN(min_samples=min_samples).fit(data)
    return algo


def compare_clusters(data, results, n_cols):
    n_rows = int(np.ceil(len(results.keys())/n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, sharey=True, sharex=True, figsize=(16, 8*n_rows))

    counter = 0
    title_names = [x for x in results.keys()]

    for i in range(0, n_rows):
        for j in range(0, n_cols):
            title = title_names[counter]
            mask = results[title] != -1

            axs[i, j].scatter(data[:, 0][~mask], data[:, 1][~mask], color='gray', alpha=0.3, s=30)
            axs[i, j].scatter(data[:, 0][mask], data[:, 1][mask], c=results[title][mask]+1, alpha=0.5, s=50, cmap="rainbow")

            axs[i, j].set_title(title_names[counter])
            counter += 1
            if counter >= len(title_names):
                break

    plt.show()


def clustering(data, grid):

    tsne_data = TSNE(n_components=2, init='random', perplexity=3).fit_transform(data)

    results = {}

    for algo in grid.keys():
        for min_samples in grid[algo]["min_samples"]:
            trained_algo = train_unsupervised_algo(data, algo_name=algo, min_samples=min_samples)
            results[f"{algo}: min_s = {min_samples}"] = trained_algo.labels_

    compare_clusters(data=tsne_data, results=results, n_cols=2)

    return results, tsne_data


def plot_group_counts(results, group):
    labels = results[group]
    h, val = np.unique(labels, return_counts=True)

    plt.figure(figsize=(15,5))
    plt.title(f"{group}:\nNumber of tokens per group")
    plt.bar(h, val)
    plt.xticks(list(range(-1, np.max(labels)+1)))
    plt.show()