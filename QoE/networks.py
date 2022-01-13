import sys

import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential


def make_nnloss_model(lookback, activ, errorf, lr):
    """
    Create the Neural network model as Loss Function
    Inputs:
        - lookback: Number of past samples used
        - activ: Activation function to use
        - errorf: loss function to use
        - lr : Learning rate used by the model
    Outputs:
        - model: The loss neural network model
    """
    model = Sequential()
    model.add(Dense(15, activation=activ, input_shape=(2,)))
    model.add(Dense(15, activation=activ))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr), loss=errorf)
    return model


def make_nn_model(lookback):
    """
    Create the Main Neural network
    Inputs:
        -lookback: Number of past samples used
    Outputs:
        -model: The neural network model
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(lookback, 1)))
    # model.add(LSTM(20, activation='relu',    return_sequences=True, input_shape=(lookback,1)))
    # model.add(LSTM(20, activation='relu'))
    model.add(Dense(1))
    # model.compile(optimizer='adam', loss=cost_func10)
    return model
