import keras
from keras.layers import (
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
)
from config.Config import Config
from keras.models import Sequential


def lstm_default(config: Config):
    model = Sequential()

    model.add(RepeatVector(n=config.hyperparameters.time_window))
    model.add(LSTM(units=256, activation=keras.activations.tanh, return_sequences=True))
    model.add(LSTM(units=256, activation=keras.activations.tanh, return_sequences=True))
    model.add(LSTM(units=256, return_sequences=True, activation=keras.activations.tanh))
    model.add(LSTM(units=256, return_sequences=True, activation=keras.activations.tanh))
    model.add(
        TimeDistributed(Dense(units=config.hyperparameters.n_channels, activation=None))
    )

    return model
