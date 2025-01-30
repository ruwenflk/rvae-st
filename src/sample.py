import os
import pandas as pd
import random

from config.Config import Config
from data.Data import Data
from models.TrainModel import TrainModel

from shared import gpu_selection
from shared import plots
from shared import utils

N_EXCERPTS_PER_PATH = 5
N_SAMPLES = 25


def get_sampled_dfs_list(
    train_model, n_samples, column_names, show_sampled=False, show_size=True
):
    pre_str = ""
    if show_sampled:
        pre_str = "Sampled: "

    post_str = ""
    if show_size:
        post_str = utils.get_length_string(train_model.config)

    list_dfs = list()
    for _ in range(n_samples):
        df = pd.DataFrame(
            train_model.model.synthesize_data().numpy()[0], columns=column_names
        )
        df = utils.change_df_column_names(df, pre_str, post_str)
        list_dfs.append(df)

    return list_dfs


def plot_original_vs_sampled(
    train_model,
    data: Data,
    config: Config,
    filename,
    n_excerpts=N_EXCERPTS_PER_PATH,
    n_samples=N_SAMPLES,
):
    list_dfs = list()
    for path in config.dataset_properties.dataset_filepaths:
        file_pathes = data._determine_file_pathes(path)

        for _ in range(n_excerpts):
            loader = data.dict_loaders[random.choice(file_pathes)]
            df_original = utils.get_random_excerpt(
                loader.df_scaled, config.hyperparameters.time_window
            )

            file_name = loader.file_path.split("/")[-1]
            df_original = utils.change_df_column_names(df_original, f"{file_name} ")

            list_dfs += list([df_original])

    list_dfs_sampled = get_sampled_dfs_list(
        train_model, n_samples, data.columns, show_sampled=True, show_size=False
    )

    list_dfs += list_dfs_sampled

    fig, axs = plots.create_subplot(len(data.columns), len(list_dfs))
    plots.plot_dfs_as_lineplots(axs, list_dfs, y_lim_default=(-1, 1))
    plots.save_fig(fig, config, filename)


if __name__ == "__main__":
    gpu_selection.autoselect_gpu()

    config = Config()
    config.select_output_config()
    config.hyperparameters.batch_size = 1

    train_model = TrainModel(config)

    data = Data(config)
    plot_original_vs_sampled(train_model, data, config, "SAMPLED")
