import tensorflow as tf
import pandas as pd

from models.VaeBase import VaeBase
from config.Config import Config
from models.PlaceboBase import PlaceboBase
from shared import logger


class TrainModel:
    def __init__(self, config):
        self.config: Config = config
        self.dict_callbacks = dict()
        self.model = self._build_model()
        self._load_weights()

    def _build_model(self):
        model_class = globals()[self.config.network_components.model_name]
        model = model_class(self.config)

        model.build(
            input_shape=[
                self.config.hyperparameters.batch_size,
                self.config.hyperparameters.time_window,
                self.config.hyperparameters.n_channels
                + self.config.hyperparameters.n_conditions,
            ]
        )
        model.summary(expand_nested=True)
        return model

    def _load_weights(self, weights_path=""):
        if weights_path == "":
            weights_path = f"{self.config.output_path}/{self.config.network_components.weights_name}.h5"
        try:
            self.model.load_weights(weights_path)
            logger.log(f"Weights loaded: {weights_path}")
        except:
            logger.log("No Weights loaded.")

    def compile_model(self):
        self.model.compile()

    def train_model(self, data):
        self.activate_earlystopping_callback()
        self.activate_tensorboard_callback()

        history = self.model.fit(
            data.get_tensor_batched_train_dataset(),
            validation_data=data.get_tensor_batched_test_dataset(),
            callbacks=[self.dict_callbacks[key] for key in self.dict_callbacks],
            epochs=self.config.hyperparameters.num_epochs,
        )

        self.df_history = pd.DataFrame.from_dict(history.history)
        self.best_loss = self.dict_callbacks["early_stopping"].best

    def save_weights(self, weights_name=None):
        if weights_name == None:
            weights_name = self.config.network_components.weights_name
        weights_path = f"{self.config.output_path}/{weights_name}.h5"
        logger.log(f"saved weights to: {weights_name}")
        self.model.save_weights(weights_path)

    def save_best_loss(self):
        best_file = open(self.config.output_path + "/best.txt", "w+")
        best_file.writelines(str(self.best_loss))
        best_file.close()

    def activate_earlystopping_callback(self):
        self.dict_callbacks["early_stopping"] = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.hyperparameters.early_stopping_monitor,
            min_delta=self.config.hyperparameters.early_stopping_min_delta,
            patience=self.config.hyperparameters.early_stopping_patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

    def activate_tensorboard_callback(self):
        self.dict_callbacks["tensorboard"] = tf.keras.callbacks.TensorBoard(
            log_dir=self.config.output_path, histogram_freq=1
        )

    def activate_checkpoint_callback(self, str_prepand=""):
        checkpoint_path = (
            self.config.output_path
            + "/checkpoints/"
            + f"{str_prepand}"
            + "-{epoch:04d}.ckpt"
        )

        self.dict_callbacks["model_checkpoint"] = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq="epoch",
        )

        self.model.save_weights(checkpoint_path.format(epoch=0))
