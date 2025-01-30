from config.Config import Config
import train
from sample import plot_original_vs_sampled
import multiprocessing as mp
from models.TrainModel import TrainModel
from data.Data import Data
from shared import utils
from shared import logger
import sys


def train_save_and_sample_long_window(
    config, weights_name, previous_weights_name, train_iteration_number
):
    weights_filepath = f"{config.output_path}/{weights_name}.h5"
    previous_weights_path = f"{config.output_path}/{previous_weights_name}.h5"
    if not utils.file_exists(weights_filepath):
        config.network_components.weights_name = weights_name
        train_model, data = train.init_trainmodel_and_data(config)
        train_model._load_weights(previous_weights_path)
        if config.hyperparameters.activate_checkpoints:
            train_model.activate_checkpoint_callback(f"{train_iteration_number}")

        train.train(train_model, config, data)

        config.hyperparameters.time_window = 1500
        train_model_long_window = TrainModel(config)

        plot_original_vs_sampled(
            train_model_long_window,
            Data(config),
            config,
            f"{train_iteration_number}-sampled",
        )
    else:
        logger.log(f"{weights_filepath} already exists. Skipping Train.")


def start_circle_train(config: Config, time_windows, alphas):
    previous_weights = "weights.h5"
    for i, (time_window, alpha) in enumerate(zip(time_windows, alphas)):
        config.hyperparameters.time_window = time_window
        config.hyperparameters.alpha = alpha
        config.hyperparameters.splitting_step_size = int(time_window / 10)
        weights_name = f"{i}-weights_{time_window}"
        p = mp.Process(
            target=train_save_and_sample_long_window,
            args=(config, weights_name, previous_weights, i),
        )
        p.start()
        p.join()
        previous_weights = weights_name


def train_forward_only(config: Config):
    w_start = 100
    w_end = 1000
    w_step = 100

    alpha_start = 5

    window_sizes = list(range(w_start, w_end + w_step, w_step))
    alphas = [w_start / w * alpha_start for w in window_sizes]

    start_circle_train(config, window_sizes, alphas)


def copy_initial_weights(config: Config):
    initial_weights_path = None
    try:
        initial_weights_path = (
            config.output_folder_path + "/" + config.modelspecific.initial_weights_path
        )

        config.create_output_folder()

        target_weights_path = (
            config.output_path + "/" + initial_weights_path.split("/")[-1]
        )

        if utils.file_exists(initial_weights_path):
            utils.copy_file(initial_weights_path, target_weights_path)
    except:
        return


if __name__ == "__main__":
    config = Config()
    if len(sys.argv) > 1 and sys.argv[1] == "continue":
        config.select_output_config()
    elif len(sys.argv) > 1 and ".conf" in sys.argv[1]:
        config.select_config(sys.argv[1], sys.argv[2])
    else:
        config.select_config()

    # copy_initial_weights(config)
    train_forward_only(config)
