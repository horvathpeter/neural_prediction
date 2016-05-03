from .base import Base

import numpy as np

import matplotlib.pyplot as plt
from neupre.misc.dataops import load_data, load_data_days
from neupre.misc.dataops import load_data_days_more
from neupre.misc.dataops import get_validation_data
from neupre.misc.builders import build_model_lstm, build_model_lstm_simple, build_model_recurrent, build_model_mlp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from neupre.instructions.neuralops import mape

plt.style.use('ggplot')

from keras.callbacks import Callback


class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.loses_batch_end = []
        self.val = []

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.loses_batch_end.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class StaticLSTM(Base):
    def run(self):

        if self.options['--onestep']:
            X_train, y_train, X_test, y_test, mean, std = load_data(path=self.options['-f'], reshape=True)
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            # model = build_model_lstm(95, 50, 25, 1)
            model = build_model_lstm_simple(inputs=95, hiddens1=50, outputs=1)
            # train
            # log = model.fit(X_train, y_train, nb_epoch=10, batch_size=100, validation_split=0.1, verbose=2)
            my_log = LossHistory()
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_data=(X_val, y_val), verbose=1,
                            callbacks=[my_log])

            # print log.history
            plt.figure(10)
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.savefig(self.options['-f'] + '_lstm_onestep_mse.png')
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            with open(self.options['-f'] + '_lstm_stats.txt', 'w'): pass
            f = open(self.options['-f'] + '_lstm_stats.txt', 'a')
            f.write("Onestep Stats\n")
            f.write("MAE : %f\n" % mean_absolute_error(y_test, y_pred))
            f.write("MSE : %f\n" % mean_squared_error(y_test, y_pred))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            plt.figure(11, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            # plt.show()
            plt.savefig(self.options['-f'] + '_lstm_onestep_pred.png')

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            f.write("MAPE : %f\n\n" % mape(y_test, y_pred))
            print("MAPE ", mape(y_test, y_pred))
            f.close()

        if self.options['--multistep']:
            # X_train, y_train, X_test, y_test, mean, std = load_data_days_more(path=self.options['-f'], reshape=True, shuffle=True)
            X_train, y_train, X_test, y_test, mean, std = load_data_days(path=self.options['-f'], reshape=True,
                                                                         shuffle=True)
            model = build_model_lstm_simple(96, 200, 96)
            # model = build_model_lstm_simple(96*5, 200, 96)
            # model = build_model_recurrent(96, 200, 96)


            # train
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)

            my_log = LossHistory()

            # log = model.fit(X_train, y_train, nb_epoch=30, batch_size=20, validation_split=0.1, verbose=2)
            log = model.fit(X_train, y_train, nb_epoch=100, batch_size=100, validation_data=(X_val, y_val), verbose=1,
                            callbacks=[my_log])

            print log.history

            plt.figure(2)
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.savefig(self.options['-f'] + '_lstm_multistep_mse.png')

            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1]))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

            f = open(self.options['-f'] + '_lstm_stats.txt', 'a')
            f.write("Multistep Stats\n")
            f.write("MAE : %f\n" % mean_absolute_error(y_test, y_pred))
            f.write("MSE : %f\n" % mean_squared_error(y_test, y_pred))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            plt.figure(3, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            # plt.show()
            plt.savefig(self.options['-f'] + '_lstm_multistep_pred.png')

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            f.write("MAPE : %f\n" % mape(y_test, y_pred))
            print("MAPE ", mape(y_test, y_pred))
            f.close()

        if self.options['--onestep96']:
            from neupre.misc.dataops import load_data_corr
            X_train, y_train, X_test, y_test = load_data_corr(reshape=True,
                                                              path='neupre/misc/data_zscore/8_ba_suma_zscore.csv',
                                                              zscore=False,
                                                              shuffle=True)
            model = build_model_lstm(6, 3, 2, 1)
            model2 = build_model_lstm_simple(6, 3, 1)
            model3 = build_model_mlp(6, 3, 1)
            model4 = build_model_recurrent(6, 3, 1)

            # train
            log = model.fit(X_train, y_train, nb_epoch=2, batch_size=10, validation_split=0.1, verbose=1)
            log = model2.fit(X_train, y_train, nb_epoch=2, batch_size=10, validation_split=0.1, verbose=1)
            log = model3.fit(X_train, y_train, nb_epoch=2, batch_size=10, validation_split=0.1, verbose=1)
            log = model4.fit(X_train, y_train, nb_epoch=2, batch_size=10, validation_split=0.1, verbose=1)

            # predict
            y_pred = model.predict(X_test)
            y_pred = model2.predict(X_test)
            y_pred = model3.predict(X_test)
            y_pred = model4.predict(X_test)

            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            plt.figure(1, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.show()

            mean = 119694.44453877833
            std = 27297.4321577972

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            print("MAPE ", mape(y_test, y_pred))

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
