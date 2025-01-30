import keras
from config.Config import Config
from keras.layers import LSTM, GRU, Conv1D, Flatten, Input
from keras.models import Sequential


def lstm_default(config: Config):
    model = Sequential()
    model.add(
        LSTM(
            units=256,
            input_shape=(
                config.hyperparameters.time_window,
                config.hyperparameters.n_channels + config.hyperparameters.n_conditions,
            ),
            return_sequences=True,
            activation=keras.activations.tanh,
        )
    )
    model.add(LSTM(units=256, return_sequences=True, activation=keras.activations.tanh))
    model.add(LSTM(units=256, activation=keras.activations.tanh, return_sequences=True))
    model.add(LSTM(units=256, activation=keras.activations.tanh))

    return model
