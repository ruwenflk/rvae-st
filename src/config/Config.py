import ast
import os
import numpy as np
from configobj import ConfigObj
from datetime import datetime

from shared import utils


# member variables must have the same names as config subsections
class Config:
    config_folder = "../config/"
    output_folder_path = "../output"
    config_file_name = "config.conf"

    list_values = ["dataset_load_condition_functions", "dataset_filepaths"]

    def __init__(self):
        self.dataset_properties = DatasetProperties()
        self.network_components = NetworkComponents()
        self.hyperparameters = HyperParameters()
        self.modelspecific = ModelSpecific()

    def select_output_config_by_args(
        self, output_path, config_name="config.conf", subsection=None
    ):
        self.output_path = output_path
        config_file = self._load_config_file(f"{output_path}/{config_name}")
        if subsection == None:
            for subsec in config_file:
                self._init_config(config_file[subsec])
                break
        else:
            conf_section = config_file[subsection]
            self._init_config(conf_section)

    def select_output_config(self):
        config_path = ""
        output_path = ""

        while not os.path.isfile(config_path):
            output_folder = f"{self.output_folder_path}{output_path}/"
            output_path = f"{output_path}/{self._select_output_path(output_folder)}"
            config_path = (
                f"{self.output_folder_path}{output_path}/{self.config_file_name}"
            )

        config = self._select_config(config_path)
        self.output_path = f"{self.output_folder_path}{output_path}"
        self._init_config(config)

    def select_config(self, config_path="", config_section_name=""):
        if len(config_path) == 0:
            config_path = self._select_config_file()
        self.config_name = config_path.split("/")[-1]
        config = self._select_config(config_path, config_section_name)
        self._set_output_path()
        self._init_config(config)

    def create_output_folder(self, output_path_addition=""):
        self.output_path = f"{self.output_path}{output_path_addition}"
        os.makedirs(self.output_path, exist_ok=True)

    def update_dimensions(self, data):
        self.hyperparameters.n_channels = data.n_channels
        self.hyperparameters.n_conditions = data.n_conditions

    def save_config_file(self):
        output_path = f"{self.output_path}/{self.config_file_name}"
        output_file = open(output_path, "w+")
        output_file.writelines(f"[{self.config_section_name}]\n")

        members = [
            "dataset_properties",
            "network_components",
            "hyperparameters",
            "modelspecific",
        ]

        for member in members:
            output_file.writelines(f"[[{member}]]\n")
            member_object = getattr(self, member)
            member_object_names = member_object.__dict__.keys()

            for object_name in member_object_names:
                value = getattr(member_object, object_name)

                if isinstance(value, np.ndarray):
                    value = [str(v) for v in value]

                if isinstance(value, list):
                    value = ",".join(value)

                if object_name in self.list_values:
                    value = f"{value},"

                output_file.writelines(f"{object_name}={value}\n")

        output_file.close()

    def _select_output_path(self, output_folder):
        folder_list = os.listdir(output_folder)
        folder_list.sort()

        output_path = utils.get_selection_value(
            "Select config file",
            utils.get_index_based_dict(folder_list, start_index=1),
        )
        return output_path

    def _set_output_path(self):
        separator = "__"
        timestamp = utils.get_timestamp()
        folder_name = f"{timestamp}{separator}{self.config_name}{separator}{self.config_section_name}"
        self.output_path = f"{self.output_folder_path}/{folder_name}"

    def _select_config_file(self):
        config_files_valid = list()

        for f in os.listdir(self.config_folder):
            if f.split(".")[1] == "conf":
                config_files_valid.append(f)

        config_name = utils.get_selection_value(
            "Select config file",
            utils.get_index_based_dict(config_files_valid, start_index=1),
        )

        return f"{self.config_folder}{config_name}"

    def _load_config_file(self, config_path):
        config_file = ConfigObj(config_path)
        return config_file

    def _select_config_section(self, config_file):
        section_name = utils.get_selection_value(
            "Select config section",
            utils.get_index_based_dict(config_file.sections, start_index=1),
        )
        return section_name

    def _select_config(self, config_path, config_section_name=""):
        config_file = self._load_config_file(config_path)
        if len(config_section_name) == 0:
            config_section_name = self._select_config_section(config_file)
        self.config_section_name = config_section_name
        return config_file[config_section_name]

    def _init_config(self, config_section):
        for subsection_name in config_section:
            subsection_dict = config_section[subsection_name]
            class_name = getattr(self, subsection_name)

            for key in subsection_dict:
                val = self._parse(subsection_dict[key])
                setattr(class_name, key, val)

    def _parse(self, val_str):
        try:
            return ast.literal_eval(val_str)
        except:
            try:
                return np.array([ast.literal_eval(v) for v in val_str])
            except:
                return val_str


# class member variables must have the same names as subsection variable names in config files
class HyperParameters:
    def __init__(self):
        self.latent_dim = 256
        self.num_epochs = 500
        self.train_test_split = 0.9
        self.post_slide_test_split = False
        self.time_window = 1000
        self.splitting_step_size = 20
        self.batch_size = 32
        self.n_channels = 0
        self.n_conditions = 0
        self.learn_rate = 0.0001
        self.alpha = 0.5
        self.beta = 0.1
        self.early_stopping_patience = 100
        self.early_stopping_min_delta = 0
        self.early_stopping_monitor = "total_loss"
        self.clipnorm = None
        # Checkpoint means that after each epoch, weights are specifically saved in a unique filename.
        self.activate_checkpoints = False


# class member variables must have the same names as subsection variable names in config files
class NetworkComponents:
    def __init__(self):
        self.decoder = "lstm_default"
        self.encoder = "lstm_default"
        self.loss_fnc = "mse_kl"
        self.model_name = "VaeBase"
        self.condition_function = "no_condition"
        self.weights_name = "weights"


# class member variables must have the same names as subsection variable names in config files
class DatasetProperties:
    def __init__(self):
        self.dataset_loader_function = ""
        self.dataset_load_condition_functions = ["load_no_condition"]
        self.dataset_filepaths = [""]
        self.scale_per_dataset = False


class ModelSpecific:
    def __init__(self):
        return
