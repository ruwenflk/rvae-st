from typing import List
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from shared import logger
from shared import utils

from data.loaders import lenze
from config.Config import Config
import os


scaler_filename = "scaler.save"


def get_scaler(scaler_path, df, min_val=-1, max_val=1):
    mm_scaler = None
    try:
        mm_scaler = joblib.load(scaler_path)
        logger.log(f"loaded scaler from {scaler_path}")
    except:
        mm_scaler = MinMaxScaler(feature_range=(min_val, max_val))
        mm_scaler.fit(df)
        joblib.dump(mm_scaler, scaler_path)
        logger.log(f"saved scaler to {scaler_path}")

    mm_scaler.set_output(transform="pandas")

    return mm_scaler


class Loader:
    fn_loader = None
    fn_conditions = None

    config: Config
    file_path: str
    scaler_path: str
    loader_function_name: str
    condition_function_name: str

    df_raw: pd.DataFrame
    df_scaled: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame

    def __init__(self, config: Config, file_path, condition_function=None):
        self.config = config
        self.file_path = file_path
        self.scaler_path = f"{self.config.output_path}/{file_path.replace('.', ',').replace('/', '|')}-{scaler_filename}"
        self.loader_function_name = config.dataset_properties.dataset_loader_function
        self.condition_function_name = condition_function
        self._set_loader_function()
        self._set_raw_data()

    def _set_loader_function(self):
        self.fn_loader = getattr(lenze, self.loader_function_name)

    def _get_df_from_csv_file_path(self):
        logger.log_headline("loading data")
        logger.log(f"loading data from path: {self.file_path}")
        if self.file_path == "":
            return pd.DataFrame()
        else:
            return pd.read_csv(self.file_path, low_memory=True)

    def _set_raw_data(self):
        df = self._get_df_from_csv_file_path()
        self.df_raw = self.fn_loader(df)

    def get_scaler(self, min_val=-1, max_val=1):
        return get_scaler(self.scaler_path, self.df_raw, min_val, max_val)

    def set_scaled_df(self, scaler: MinMaxScaler):
        self.df_scaled = scaler.transform(self.df_raw)

    def set_train_test_split(self, factor):
        split = int(self.df_raw.shape[0] * factor)
        self.df_train = self.df_scaled.iloc[:split, :]
        self.df_test = self.df_scaled.iloc[split:, :]

    def get_condition(self):
        return self.fn_conditions(self.config)


class Data:
    scaler_path: str
    n_channels: int
    n_conditions: int

    conditional: bool

    def __init__(self, config: Config):
        self.dict_loaders: dict[str, Loader] = dict()
        self.config = config
        self.scaler_path = f"{self.config.output_path}/{scaler_filename}"

        self._initialize_loaders()
        self._determine_dimensions()
        self._set_conditional_flag()

        if self.config.dataset_properties.scale_per_dataset:
            self._scale_per_dataset()
        else:
            self._scale()

        self._train_test_split()

    def get_sliding_windows(self, set_name, remove_conditions=False):
        def fn_set_train_loader(loader):
            if set_name == "train":
                return loader.df_train
            elif set_name == "test":
                return loader.df_test
            elif set_name == "scaled":
                return loader.df_scaled
            else:
                logger.log("Error. No Subset selected for Dataset")

        arr_sliding_windows = self._create_sliding_windows(fn_set_train_loader)
        if remove_conditions:
            arr_sliding_windows = self._remove_conditions(arr_sliding_windows)

        return arr_sliding_windows

    def get_tensor_batched_train_dataset(self):
        train_set = self.get_sliding_windows("train")

        if self.config.hyperparameters.post_slide_test_split:
            train_set, _ = utils.train_test_split(
                train_set, self.config.hyperparameters.train_test_split
            )

        return self._from_tensor_slices(train_set)

    def get_tensor_batched_test_dataset(self):
        if self.config.hyperparameters.post_slide_test_split:
            _, test_set = utils.train_test_split(
                self.get_sliding_windows("train"),
                self.config.hyperparameters.train_test_split,
            )
            return self._from_tensor_slices(test_set)

        return self._from_tensor_slices(self.get_sliding_windows("test"))

    def _determine_dimensions(self):
        for key in self.dict_loaders:
            loader = self.dict_loaders[key]
            self.columns = loader.df_raw.columns  # can be removed
            self.n_channels = loader.df_raw.shape[1]
            try:
                # if loaders differ in number of conditions, this will cause a bug
                self.n_conditions = loader.get_condition().shape[1]
                logger.log(f"{self.n_conditions} conditions used.")
            except:
                logger.log(f"Zero conditions used.")
                self.n_conditions = 0
            break

    def _set_class_loader(self, file_path, condition_index):
        class_loader = Loader(
            self.config,
            file_path,
            self.config.dataset_properties.dataset_load_condition_functions[
                condition_index
            ],
        )
        self.dict_loaders[file_path] = class_loader

    def _determine_file_pathes(self, path):
        file_pathes = list()
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                file_pathes.append(os.path.join(path, file_name))
        else:
            file_pathes.append(path)

        return file_pathes

    def _initialize_loaders(self):
        for i, path in enumerate(self.config.dataset_properties.dataset_filepaths):
            for file_path in self._determine_file_pathes(path):
                self._set_class_loader(file_path, i)

    def _set_conditional_flag(self):
        self.conditional = False

    def _scale(self, min_val=-1, max_val=1):
        list_dfs = [self.dict_loaders[key].df_raw for key in self.dict_loaders]
        df_all = pd.concat(list_dfs)

        mm_scaler = get_scaler(self.scaler_path, df_all, min_val, max_val)

        for key in self.dict_loaders:
            self.dict_loaders[key].set_scaled_df(mm_scaler)

    def _scale_per_dataset(self, min_val=-1, max_val=1):
        for key in self.dict_loaders:
            dict_loader = self.dict_loaders[key]
            mm_scaler = dict_loader.get_scaler(min_val, max_val)
            dict_loader.set_scaled_df(mm_scaler)

    def _train_test_split(self):
        factor = self.config.hyperparameters.train_test_split
        if self.config.hyperparameters.post_slide_test_split:
            factor = 1

        for key in self.dict_loaders:
            self.dict_loaders[key].set_train_test_split(factor)

    def _create_sliding_windows(self, fn_set_loader):
        # example [1,2,3,4,5,6,7] -> [[1,2,3], [3,4,5], [5,6,7]] with splitting_step_size=2 and window_size=3
        df_condition = None
        X = list()
        for key in self.dict_loaders:
            loader = self.dict_loaders[key]
            df_arr = fn_set_loader(loader)

            if self.conditional:
                df_condition = loader.get_condition()

            X += utils.create_sliding_windows(
                df_arr,
                self.config.hyperparameters.time_window,
                self.config.hyperparameters.splitting_step_size,
                df_condition,
            )

        return np.array(X)

    def _remove_conditions(self, arr):
        if self.conditional and arr.shape[2] == self.n_channels + self.n_conditions:
            arr = arr[:, :, : -self.n_conditions]
        return arr

    def _from_tensor_slices(self, x):
        y = self._remove_conditions(x)

        return (
            tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(x.shape[0], reshuffle_each_iteration=True)
            .batch(
                batch_size=self.config.hyperparameters.batch_size, drop_remainder=True
            )
        )

    def _from_tensor_slices_categorical(
        self, x, label: int
    ):  # ist discriminator score spezifisch, würde ich wahrscheinlich aus der Klasse herausnehmen
        y = self._remove_conditions(x)
        shape = (y.shape[0], 1)
        y = np.ones(shape=shape) * label
        return (
            tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(x.shape[0], reshuffle_each_iteration=True)
            .batch(
                batch_size=self.config.hyperparameters.batch_size, drop_remainder=True
            )
        )
