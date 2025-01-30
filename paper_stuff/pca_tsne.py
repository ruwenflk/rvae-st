import sys, os
import numpy as np

module_path = os.path.abspath(os.path.join(".."))
sys.path.append(module_path + "/src")

from config.Config import Config
from models.TrainModel import TrainModel
from shared import utils
import pandas as pd
from data.Data import Data
from shared import plots

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from shared import gpu_selection
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def get_shuffled_original_chunks(data, n_samples, subset):
    arr_orig_chunks = data.get_sliding_windows(subset, remove_conditions=True)
    np.random.shuffle(arr_orig_chunks)
    arr_orig_chunks = arr_orig_chunks[:n_samples]
    return arr_orig_chunks


def reduce_chunks(arr, config):
    return arr.reshape(-1, config.hyperparameters.time_window)


def get_prepared_reduced_chunks(model, data, config, n_samples, subset):
    arr_orig_chunks = get_shuffled_original_chunks(data, n_samples, subset)
    arr_synthetic_chunks = utils.get_synthetic_chunks(model, len(arr_orig_chunks))

    arr_orig_chunks_reduced = reduce_chunks(arr_orig_chunks, config)
    arr_synthetic_chunks_reduced = reduce_chunks(arr_synthetic_chunks, config)

    return arr_orig_chunks_reduced, arr_synthetic_chunks_reduced


def get_data_and_model(config):
    data = Data(config)
    model = TrainModel(config).model
    return data, model


def plot_pca_and_tsne(pca_real, pca_synth, tsne_results):
    fig = plt.figure(constrained_layout=True, figsize=(9, 3))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # TSNE scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title("pca", fontsize=10, color="black", pad=10)
    markersize = 5
    # PCA scatter plot
    plt.scatter(
        pca_real.iloc[:, 0].values,
        pca_real.iloc[:, 1].values,
        s=markersize,
        c="blue",
        alpha=0.4,
        label="Original",
    )
    plt.scatter(
        pca_synth.iloc[:, 0],
        pca_synth.iloc[:, 1],
        s=markersize,
        c="red",
        alpha=0.3,
        label="Synthetic",
    )
    ax.legend(fontsize=8)

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title("t-sne", fontsize=10, color="black", pad=10)

    sample_size = len(tsne_results) // 2
    plt.scatter(
        tsne_results.iloc[:sample_size, 0].values,
        tsne_results.iloc[:sample_size, 1].values,
        s=markersize,
        c="blue",
        alpha=0.4,
        label="Original",
    )
    plt.scatter(
        tsne_results.iloc[sample_size:, 0],
        tsne_results.iloc[sample_size:, 1],
        s=markersize,
        c="red",
        alpha=0.3,
        label="Synthetic",
    )

    ax2.legend(fontsize=8)

    # fig.suptitle(
    #    "Validating synthetic vs real data diversity and distributions",
    #    fontsize=16,
    #    color="grey",
    # )
    dataset_name = config.output_path.split("/")[-1]
    seq = config.hyperparameters.time_window
    plt.savefig(f"{config.output_path}/{dataset_name}_{seq}_both.png")


def get_pca_and_tsne_results(chunks_original, chunks_synthetic):
    pca = PCA(n_components=2)
    pca.fit(chunks_original)

    pca_real = pd.DataFrame(pca.transform(chunks_original))
    pca_synth = pd.DataFrame(pca.transform(chunks_synthetic))

    tsne = TSNE(n_components=2, n_iter=300)
    data_concat = np.concatenate((chunks_original, chunks_synthetic), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_concat))

    return pca_real, pca_synth, tsne_results


def plot_pca(config, pca_real, pca_synth):
    fig = plt.figure(constrained_layout=True, figsize=(4.5, 3))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)

    # TSNE scatter plot
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title("PCA", fontsize=14, color="black", pad=10)

    # fig, axs = plots.create_subplot(1, 1, row_size=9, columns_size=3, font_size=10)
    # ax = axs
    # ax.set_title("pca", fontsize=10, color="black", pad=10)

    markersize = 5
    # PCA scatter plot
    ax.scatter(
        pca_real.iloc[:, 0].values,
        pca_real.iloc[:, 1].values,
        s=markersize,
        c="blue",
        alpha=0.4,
        label="Original",
    )
    ax.scatter(
        pca_synth.iloc[:, 0],
        pca_synth.iloc[:, 1],
        s=markersize,
        c="red",
        alpha=0.3,
        label="Synthetic",
    )
    ax.legend(fontsize=8)
    ax.set_xticks([])  # Entfernt X-Ticks
    ax.set_yticks([])  # Entfernt Y-Ticks

    dataset_name = config.output_path.split("/")[-1]
    seq = config.hyperparameters.time_window
    # fig.savefig(f"{config.output_path}/{dataset_name}_{seq}_pca.png")
    return fig


def plot_tsne(config, tsne_results):
    fig = plt.figure(constrained_layout=True, figsize=(4.5, 3))
    spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    markersize = 5

    # PCA scatter plot
    ax2 = fig.add_subplot(spec[0, 0])
    ax2.set_title("t-SNE", fontsize=14, color="black", pad=10)

    sample_size = len(tsne_results) // 2
    ax2.scatter(
        tsne_results.iloc[:sample_size, 0].values,
        tsne_results.iloc[:sample_size, 1].values,
        s=markersize,
        c="blue",
        alpha=0.4,
        label="Original",
    )
    ax2.scatter(
        tsne_results.iloc[sample_size:, 0],
        tsne_results.iloc[sample_size:, 1],
        s=markersize,
        c="red",
        alpha=0.3,
        label="Synthetic",
    )

    ax2.legend(fontsize=8)
    ax2.set_xticks([])  # Entfernt X-Ticks
    ax2.set_yticks([])  # Entfernt Y-Ticks

    dataset_name = config.output_path.split("/")[-1]
    seq = config.hyperparameters.time_window
    # fig.savefig(f"{config.output_path}/{dataset_name}_{seq}_tsne.png")
    return fig


if __name__ == "__main__":
    # gpu_selection.select_gpu_with_lowest_memory()
    config = Config()
    if len(sys.argv) == 3:
        output_path, config_name = sys.argv[1].rsplit("/", 1)
        config.select_output_config_by_args(
            output_path, config_name, subsection=sys.argv[2]
        )
    else:
        config = Config()
        config.select_output_config()

    n_samples = 250
    subset = "train"

    data, model = get_data_and_model(config)

    chunks_original, chunks_synthetic = get_prepared_reduced_chunks(
        model, data, config, n_samples, subset
    )

    pca_real, pca_synth, tsne_results = get_pca_and_tsne_results(
        chunks_original, chunks_synthetic
    )

    # plot_pca_and_tsne(pca_real, pca_synth, tsne_results)
    plot_pca(pca_real, pca_synth)
    plot_tsne(tsne_results)
