from config.Config import Config
import sys
from models.TrainModel import TrainModel
from shared import utils
import pandas as pd
import numpy as np
from tqdm import tqdm
from shared.utils import logger
from shared import gpu_selection

if __name__ == "__main__":
    config_elbo_model = Config()
    config_sampling_model = Config()
    gpu_selection.autoselect_gpu()
    print("###################")
    print(sys.argv)

    if len(sys.argv) == 5:
        output_path, config_name = sys.argv[1].rsplit("/", 1)
        config_elbo_model.select_output_config_by_args(
            output_path, config_name, subsection=sys.argv[2]
        )

        output_path, config_name = sys.argv[3].rsplit("/", 1)
        config_sampling_model.select_output_config_by_args(
            output_path, config_name, subsection=sys.argv[4]
        )
    else:
        config_elbo_model.select_output_config()
        config_sampling_model.select_output_config()

    model_elbo = TrainModel(config_elbo_model).model
    model_sampling = TrainModel(config_sampling_model).model

    list_elbos = list()
    n_samples = 1000
    arr_samples = utils.get_synthetic_chunks(model_sampling, n_samples)
    logger.log("Calculating ELBOs...")
    for sample in tqdm(arr_samples):
        df_sampled = pd.DataFrame(sample)

        sampled_splitted = np.array(
            utils.create_sliding_windows(
                df_sampled,
                config_elbo_model.hyperparameters.time_window,
                config_elbo_model.hyperparameters.splitting_step_size,
            )
        )

        elbo = model_elbo.call_and_loss(sampled_splitted, sampled_splitted)[0].numpy()
        # list_elbos.append(elbo)
        list_elbos += list(elbo)

    from shared.elbo_calculator import get_elbo_reduced

    list_elbos = np.asarray(list_elbos)
    list_elbos = list_elbos[np.isfinite(list_elbos)]
    list_elbos = get_elbo_reduced(
        list_elbos,
        config_elbo_model.hyperparameters.time_window,
        config_elbo_model.hyperparameters.n_channels,
        config_elbo_model.hyperparameters.alpha,
        config_elbo_model.hyperparameters.beta,
    )

    elbo_mean = np.mean(list_elbos)
    elbo_std = np.std(list_elbos)
    print(f"{elbo_mean} +- {elbo_std}")

    with open(
        config_sampling_model.output_path
        + f"/vae_{config_sampling_model.hyperparameters.time_window}.txt",
        "w",
    ) as f:
        f.write(f"{elbo_mean}+-{elbo_std}")
        f.close()

    with open(
        config_sampling_model.output_path
        + f"/vae_significance_{config_sampling_model.hyperparameters.time_window}.txt",
        "w",
    ) as f:
        for val in list_elbos:
            f.write(f"{val}\n")
        f.close()
