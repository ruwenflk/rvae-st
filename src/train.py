from config.Config import Config
from data.Data import Data
from models.TrainModel import TrainModel
from shared import gpu_selection
from shared import plots
from shared import utils
import sample

import pandas as pd


def plot_history(config: Config, df_history: pd.DataFrame):
    fig, axs = plots.create_subplot(3, 2)
    plots.plot_df_as_lineplot(
        axs[:, 0], pd.DataFrame(df_history, columns=df_history.columns[:3])
    )
    plots.plot_df_as_lineplot(
        axs[:, 1], pd.DataFrame(df_history, columns=df_history.columns[-3:])
    )
    plots.save_fig(fig, config, "history")


def init_trainmodel_and_data(config: Config, output_path_addition=""):
    gpu_selection.autoselect_gpu()

    config.create_output_folder(output_path_addition)
    data = Data(config)
    config.update_dimensions(data)
    config.save_config_file()

    train_model = TrainModel(config)
    return train_model, data


def init_and_train(config, output_path_addition=""):
    train_model, data = init_trainmodel_and_data(config, output_path_addition)

    if config.hyperparameters.activate_checkpoints:
        train_model.activate_checkpoint_callback()

    train_model = train(train_model, config, data)
    return train_model


def train(train_model: TrainModel, config: Config, data):
    train_model.compile_model()
    train_model.train_model(data)
    train_model.save_weights()
    train_model.save_best_loss()

    utils.save_history_as_csv(config, train_model.df_history)
    plot_history(config, train_model.df_history)
    sample.plot_original_vs_sampled(train_model, data, config, "orig-vs-sample")
    return train_model


def train_and_save_custom_weights(
    train_model: TrainModel, config: Config, data, weights_name: str
):
    train_model = train(train_model, config, data)
    train_model.save_weights(weights_name)
    return train_model


if __name__ == "__main__":
    config = Config()
    config.select_config()
    init_and_train(config)
