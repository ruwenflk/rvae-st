import collections
from datetime import datetime
from shared import logger
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm


def get_selection_index(task_headline, dict):
    logger.log_headline(task_headline)

    dictionary = collections.OrderedDict(sorted(dict.items()))

    for key in dictionary:
        logger.log("{0}: {1}".format(key, dictionary[key]))

    selected_index = int(input("ID: "))  # TODO exception handling

    return selected_index


def get_selection_value(task_headline, dict):
    index = get_selection_index(task_headline, dict)
    return dict[index]


def get_index_based_dict(
    iteration_object, start_index, callback_function=lambda val: val
):
    dictionary = {}

    for index, key in enumerate(iteration_object):
        dictionary[index + start_index] = callback_function(key)

    return dictionary


def get_timestamp():
    separator = "__"
    return datetime.now().strftime("%Y-%m-%d{}%H:%M:%S".format(separator))


def get_min_from_df(df):
    return np.min(df.min())


def get_max_from_df(df):
    return np.min(df.max())


def get_min_max_from_dfs(dfs_list):
    max_vals = list()
    min_vals = list()
    for df in dfs_list:
        min_vals.append(get_min_from_df(df))
        max_vals.append(get_max_from_df(df))

    return min(min_vals), max(max_vals)


def get_min_max_by_cols_as_dict(list_dfs, col_names):
    dict_col_lims = dict()
    for col_name in col_names:
        (min_val, max_val) = get_min_max_from_dfs([df[col_name] for df in list_dfs])
        dict_col_lims[col_name] = (min_val, max_val)
    return dict_col_lims


def get_random_excerpt(df, time_window):
    start = np.random.randint(df.shape[0] - time_window)
    excerpt = np.array(df.to_numpy()[start : start + time_window])
    excerpt = pd.DataFrame(excerpt, columns=df.columns)
    return excerpt


def change_df_column_names(df, pre_str="", post_str=""):
    df.columns = [f"{pre_str}" + name + f"{post_str}" for name in df.columns]
    return df


def get_length_string(config):
    return f", length={config.hyperparameters.time_window}"


def get_df_columns_from_np_array(arr):
    return [str(i) for i in range(arr.shape[1])]


def create_sliding_windows(df_arr, time_window, step, df_condition=None):
    np_arr = df_arr.to_numpy()
    X = list()
    data_length = np_arr.shape[0]
    for index_start in np.arange(0, data_length, step):
        index_end = index_start + time_window
        if index_end > data_length:
            break

        sample = np_arr[index_start:index_end, :]
        if df_condition is not None:
            sample = np.concatenate((sample, df_condition.to_numpy()), axis=1)
        X.append(sample)
    return X


def get_weight_paths(output_path, early_stopping_patience):
    ckpt_path = f"{output_path}/checkpoints/"
    paths = os.listdir(ckpt_path)
    paths = [f for f in paths if "index" in f]
    paths = [os.path.join(ckpt_path, f) for f in paths]  # add path to each file
    paths.sort(key=lambda x: os.path.getmtime(x))
    paths = [f.split(".index")[0] for f in paths]

    filtered_pathes = list()
    training_restarts = list()
    for i in range(30):
        str_start = f"{i}-"
        tmp_paths = [
            f for f in paths if f.split("/")[-1][: len(str_start)] == str_start
        ]
        tmp_paths = tmp_paths[:-early_stopping_patience]
        filtered_pathes += tmp_paths
        if len(tmp_paths) > 0:
            training_restarts += (len(tmp_paths) - 1) * [False] + [True]

    df_paths = pd.DataFrame(filtered_pathes, columns=["pathes"])
    df_restarts = pd.DataFrame(training_restarts, columns=["restarts"])
    return pd.concat([df_paths, df_restarts], axis=1)


def train_test_split(arr, split_factor):
    split_index = int(split_factor * len(arr))
    arr_2 = arr[split_index:]
    arr = arr[:split_index]
    return arr, arr_2


def get_synthetic_chunks(trained_model, n_samples):
    list_batched_samples = list()
    batch_size = len(trained_model.synthesize_data().numpy())
    n_iterations = int(np.ceil(n_samples / batch_size))

    logger.log("Sampling...")
    for _ in tqdm(range(n_iterations)):
        batched_samples = trained_model.synthesize_data().numpy()
        list_batched_samples.append(batched_samples)

    arr_chunks = np.concatenate(np.asarray(list_batched_samples))
    arr_chunks = arr_chunks[:n_samples]
    return arr_chunks


def file_exists(filepath):
    return os.path.isfile(filepath)


def copy_file(src, dst):
    shutil.copyfile(src, dst)


def save_history_as_csv(config, df_history):
    output_folder = f"{config.output_path}/history"
    os.makedirs(output_folder, exist_ok=True)
    history_path = f"{output_folder}/history-{get_timestamp()}.csv"
    df_history.to_csv(history_path)
