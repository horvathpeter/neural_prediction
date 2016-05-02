import time

from keras.layers.core import Dense
from keras.layers.core import Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential


def build_model_mlp(inputs, hiddens, outputs):
    model = Sequential()
    model.add(Dense(
        output_dim=hiddens,
        input_shape=(inputs,),
        activation='tanh'
    ))

    model.add(Dense(
        output_dim=outputs,
        activation='linear'
    ))
    start = time.time()
    print ("Compiling...")
    model.compile(loss='mse', optimizer='SGD')
    print ("Compiled and took ", time.time() - start, "seconds")
    return model


def build_model_recurrent(inputs, hiddens1, outputs):
    model = Sequential()
    model.add(SimpleRNN(
        output_dim=hiddens1,
        input_shape=(inputs, 1),
        return_sequences=False,
        activation='tanh'
    ))

    model.add(Dense(
        output_dim=outputs,
        activation='linear'
    ))
    print ("Compiling...")
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ("Compiled and took ", time.time() - start, " seconds")
    return model


def build_model_lstm_simple(inputs, hiddens1, outputs):
    model = Sequential()
    model.add(LSTM(
        output_dim=hiddens1,
        input_shape=(inputs, 1),
        return_sequences=False,
        activation='tanh'
    ))

    model.add(Dense(
        output_dim=outputs,
        activation='linear'
    ))
    print ("Compiling...")
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print ("Compiled and took ", time.time() - start, " seconds")
    return model


def build_model_lstm(inputs, hiddens1, hiddens2, outputs):
    model = Sequential()

    model.add(LSTM(
        output_dim=hiddens1,
        input_shape=(inputs, 1),
        return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(
        hiddens2,
        return_sequences=False))
    # model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=outputs))
    model.add(Activation("linear"))

    print ("Compiling...")
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compiled and took ", time.time() - start, " seconds")
    return model
