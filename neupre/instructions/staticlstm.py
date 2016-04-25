# from keras.callbacks import Callback
# from keras.layers.core import Dense, Activation
# from keras.layers.core import Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential

# from instructions.neuralops import mape, mapke


from .base import Base

import numpy as np
import matplotlib.pyplot as plt


# class LossHistory(Callback):
#     def __init__(self):
#         super(LossHistory, self).__init__()
#         self.loses_batch_begin = []
#         self.loses_batch_end = []
#
#     def on_epoch_begin(self, epoch, logs={}):
#         pass
#
#     def on_epoch_end(self, epoch, logs={}):
#         pass
#
#     def on_batch_begin(self, batch, logs={}):
#         self.loses_batch_begin.append(logs.get('loss'))
#
#     def on_batch_end(self, batch, logs={}):
#         self.loses_batch_end.append(logs.get('loss'))
#
#     def on_train_begin(self, logs={}):
#         pass
#
#     def on_train_end(self, logs={}):
#         pass


class StaticLSTM(Base):
    def run(self):
        from neupre.misc.dataops import load_data, load_data_days
        from neupre.misc.builders import build_model_lstm
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if self.options['--onestep']:
            X_train, y_train, X_test, y_test = load_data(reshape=True)
            # X_train, y_train, X_validation, y_validation = get_validation_data(X_train, y_train, 0.1)
            model = build_model_lstm(95, 50, 50, 1)
            # train
            history = model.fit(X_train, y_train, nb_epoch=1, batch_size=100, validation_split=0.1, verbose=1)
            # history = net.fit(X_train, y_train, batch_size=100, nb_epoch=1, callbacks=[lossHistory],
            #                   validation_data=(X_validation, y_validation))
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            plt.figure(0, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.show()

        if self.options['--multistep']:
            X_train, y_train, X_test, y_test = load_data_days(reshape=True)
            model = build_model_lstm(96, 50, 50, 96)
            # train
            history = model.fit(X_train, y_train, nb_epoch=20, batch_size=1, validation_split=0.1, verbose=1)
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1]))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            plt.figure(0, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.show()


# def build_96():
#     model = Sequential()
#
#     model.add(LSTM(
#         output_dim=50,
#         input_shape=(96, 1),
#         return_sequences=True))
#     model.add(Dropout(0.2))
#
#     model.add(LSTM(
#         50,
#         return_sequences=False
#     ))
#     model.add(Dropout(0.2))
#
#     model.add(Dense(
#         output_dim=96
#     ))
#     model.add(Activation("linear"))
#     print ("Compiling...")
#     start = time.time()
#     model.compile(loss="mse", optimizer="rmsprop")
#     print ("Compilation Time : ", time.time() - start)
#     return model
#
#
# def run96():
#     plt.style.use('ggplot')
#     X_train, y_train, X_test, y_test = load_data_days()
#     net96 = build_96()
#     history = net96.fit(X_train, y_train, validation_split=0.1, nb_epoch=10, batch_size=10)
#     p96 = net96.predict(X_test)
#
#     y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))
#     p96 = np.reshape(p96, (p96.shape[0] * p96.shape[1]))
#
#     print("MAE  ", mean_absolute_error(y_test, p96))
#     print("MSE  ", mean_squared_error(y_test, p96))
#
#     plt.figure(0, figsize=(20, 3))
#     plt.plot(y_test)
#     plt.plot(p96)
#
#     plt.figure(1)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#
#     plt.figure(2)
#     plt.plot(history_['loss'])
#     plt.plot(history_['val_loss'])
