import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, GRU, LSTM
import numpy as np
from sklearn.metrics import accuracy_score

from config.Config import Config
from data.Data import Data
from models.TrainModel import TrainModel
from shared import utils
import multiprocessing as mp
import sys


class DiscriminativeMetrics(Model):
    def __init__(self, config, hidden_dim=2, num_layers=1):
        super(DiscriminativeMetrics, self).__init__()
        self.discrimator = self.create_discriminator(config, hidden_dim, num_layers)
        self.loss_fnc = tf.keras.losses.BinaryCrossentropy()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.total_val_loss_tracker = keras.metrics.Mean(name="val_total_loss")

    def create_discriminator(self, config, hidden_dim, num_layers):
        discriminator = tf.keras.models.Sequential()

        for i in range(num_layers):
            return_sequences = i != (num_layers - 1)
            input_shape = (
                (
                    config.hyperparameters.time_window,
                    config.hyperparameters.n_channels,
                )
                if i == 0
                else discriminator.layers[-1].output_shape
            )

            discriminator.add(
                GRU(
                    units=hidden_dim,
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                )
            )

        discriminator.add(Dense(units=1, activation="sigmoid"))

        return discriminator

    def call(self, x):
        return self.discrimator(x)

    def train_step(self, data):
        sequence, true_label = data
        with tf.GradientTape() as tape:
            pred_label = self.call(sequence)
            total_loss = self.loss_fnc(true_label, pred_label)
            self._apply_grads(total_loss, tape)
            return self._update_train_trackers(total_loss)

    def test_step(self, data):
        sequence, true_label = data
        pred_label = self.call(sequence)
        total_loss = self.loss_fnc(true_label, pred_label)
        return self._update_test_trackers(total_loss)

    def _apply_grads(self, total_loss, tape):
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

    def _update_train_trackers(self, total_loss):
        self.total_loss_tracker.update_state(total_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
        }

    def _update_test_trackers(self, total_loss):
        self.total_val_loss_tracker.update_state(total_loss)
        return {
            "total_loss": self.total_val_loss_tracker.result(),
        }

    def compile(self):
        super(DiscriminativeMetrics, self).compile(
            optimizer=optimizers.Adam(),
            loss=self.loss_fnc,
        )


def get_discrim_model(config, hidden_dim=1, num_layers=1):
    discriminative_model = DiscriminativeMetrics(config, hidden_dim, num_layers)
    discriminative_model.build(
        input_shape=[
            config.hyperparameters.batch_size,
            config.hyperparameters.time_window,
            config.hyperparameters.n_channels,
        ]
    )
    discriminative_model.compile()
    discriminative_model.summary(expand_nested=True)
    return discriminative_model


def get_original_dataset_chunked(data, config):
    arr_chunks = data.get_sliding_windows("train", remove_conditions=True)
    arr_train, arr_test = utils.train_test_split(
        arr_chunks, config.hyperparameters.train_test_split
    )
    return arr_train, arr_test


def tensor_slice_dataset(arr_train, arr_test, data: Data):
    train_dataset = data._from_tensor_slices_categorical(arr_train, label=1)
    test_dataset = data._from_tensor_slices_categorical(arr_test, label=1)

    return (
        train_dataset,
        test_dataset,
        arr_train,
        arr_test,
    )


def get_synthetic_dataset(data, config, n_samples_train, n_samples_val, n_samples_test):
    train_model = TrainModel(config)

    fake_arr_train = utils.get_synthetic_chunks(train_model.model, n_samples_train)
    fake_arr_val = utils.get_synthetic_chunks(train_model.model, n_samples_val)
    fake_test_samples = utils.get_synthetic_chunks(train_model.model, n_samples_test)

    fake_train_samples_tensored = data._from_tensor_slices_categorical(fake_arr_train, label=0)
    fake_val_samples_tensored = data._from_tensor_slices_categorical(fake_arr_val, label=0)
    fake_test_samples_tensored = data._from_tensor_slices_categorical(fake_test_samples, label=0)

    return fake_train_samples_tensored, fake_val_samples_tensored, fake_test_samples_tensored, fake_test_samples


# this is for testing purposes for faster development
def load_config(output_path):
    config = Config()
    config.select_output_config_by_args(output_path)
    return config


def get_earlystopping_callback():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_total_loss",
        min_delta=0,
        patience=50,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )


def train_model_and_save_results_to_file(arr_train, arr_val, true_test_samples, data, config):
    n_samples_train = len(arr_train)
    n_samples_val = len(arr_val)
    n_samples_test = len(true_test_samples)
    from shared import gpu_selection

    gpu_selection.select_gpu_with_lowest_memory()

    true_train_samples_tensored = data._from_tensor_slices_categorical(arr_train, label=1)
    true_val_samples_tensored = data._from_tensor_slices_categorical(arr_val, label=1)
    true_test_samples_tensored = data._from_tensor_slices_categorical(true_test_samples, label=1)

    (
        fake_train_samples_tensored,
        fake_val_samples_tensored,
        fake_test_samples_tensored,
        fake_test_samples,
    ) = get_synthetic_dataset(data, config, n_samples_train, n_samples_val, n_samples_test)

    print(f"number of samples: {n_samples_train}")

    combined_train_dataset = true_train_samples_tensored.concatenate(fake_train_samples_tensored)
    """
    .shuffle(
        config.hyperparameters.batch_size, reshuffle_each_iteration=True
    )
    """

    combined_val_dataset = true_val_samples_tensored.concatenate(fake_val_samples_tensored)

    discriminative_model = get_discrim_model(config, hidden_dim, num_layers=1)
    history = discriminative_model.fit(
        combined_train_dataset,
        validation_data=combined_val_dataset,
        epochs=2000,
        callbacks=[get_earlystopping_callback()],
    )

    results_test_original = discriminative_model(true_test_samples).numpy()
    results_test_sampled = discriminative_model(fake_test_samples).numpy()

    true_labels = np.ones(len(true_test_samples))
    fake_labels = np.zeros(len(fake_test_samples))

    test_original_labels = results_test_original > 0.5
    test_fake_labels = results_test_sampled > 0.5

    accuracy = accuracy_score(
        np.concatenate((true_labels, fake_labels)),
        np.concatenate((test_original_labels, test_fake_labels)),
    )
    discriminative_score = np.abs(0.5 - accuracy)
    print(discriminative_score)

    with open(
        config.output_path
        + f"/discriminative_score_{config.hyperparameters.time_window}.txt",
        "a+",
    ) as f:
        f.write(f"{discriminative_score}\n")
        f.close()


if __name__ == "__main__":
    config = Config()
    if len(sys.argv) > 1 and ".conf" in sys.argv[1]:
        output_path, config_name = sys.argv[1].rsplit("/", 1)
        config.select_output_config_by_args(
            output_path, config_name, subsection=sys.argv[2]
        )
    else:
        config.select_output_config()

    data = Data(config)
    config.update_dimensions(data)
    hidden_dim = int(data.n_channels / 2)

    with open(
        config.output_path
        + f"/discriminative_score_{config.hyperparameters.time_window}.txt",
        "w",
    ) as f:
        f.close()

    score_list = list()
    #arr_train, arr_test = get_original_dataset_chunked(data, config)

    n_samples_train = 2000
    n_samples_val = 500
    n_samples_test = 500

    """np.random.shuffle(arr_train)
    arr_train = arr_train[:n_samples_train]
    arr_val = arr_train[-n_samples_val:]
    arr_train = arr_train[:n_samples_train-n_samples_val]

    np.random.shuffle(arr_test)
    arr_test = arr_test[:n_samples_test]"""
    for _ in range(15):
        arr_train, arr_test = get_original_dataset_chunked(data, config)
        np.random.shuffle(arr_train)
        arr_train = arr_train[:n_samples_train]
        arr_val = arr_train[-n_samples_val:]
        arr_train = arr_train[:n_samples_train-n_samples_val]

        np.random.shuffle(arr_test)
        arr_test = arr_test[:n_samples_test]
        p = mp.Process(
            target=train_model_and_save_results_to_file,
            args=(arr_train, arr_val, arr_test, data, config),
        )
        p.start()
        p.join()

    with open(
        config.output_path
        + f"/discriminative_score_{config.hyperparameters.time_window}.txt",
        "r+",
    ) as f:
        arr_score_list = np.asarray(f.readlines()).astype(np.float32)
        f.write(f"{arr_score_list.mean()}+-{arr_score_list.std()}")
        f.close()
