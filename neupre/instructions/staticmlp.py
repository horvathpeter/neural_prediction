import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from os import listdir
from os.path import isfile, join

from neupre.misc.dataops import load_data, load_data_days, load_data_corr
from neupre.misc.dataops import load_data_days_more
from neupre.misc.dataops import get_validation_data
from neupre.misc.builders import build_model_mlp
from neupre.instructions.neuralops import mape
from .base import Base

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


class StaticMlp(Base):
    def run(self):

        if self.options['--onestep']:
            X_train, y_train, X_test, y_test, mean, std = load_data(path=self.options['-f'], sequence_length=96,
                                                                    shuffle=False)
            model = build_model_mlp(inputs=95, hiddens=50, outputs=1)
            # train
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            my_log = LossHistory()
            # log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_split=0.1, verbose=2)
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_data=(X_val, y_val), verbose=2, callbacks=[my_log])
            # predict
            # print log.history
            plt.figure(0)
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.savefig(self.options['-f']+'_mlp_onestep_mse.png')

            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            with open(self.options['-f']+'_mlp_stats.txt', 'w'): pass
            f = open(self.options['-f']+'_mlp_stats.txt', 'a')
            f.write("Onestep Stats\n")
            f.write("MAE : %f\n" % metrics.mean_absolute_error(y_test, y_pred))
            f.write("MSE : %f\n" % metrics.mean_squared_error(y_test, y_pred))

            print("MAE  ", metrics.mean_absolute_error(y_test, y_pred))
            print("MSE  ", metrics.mean_squared_error(y_test, y_pred))

            plt.figure(1, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            # plt.show()
            plt.savefig(self.options['-f'] + '_mlp_onestep_pred.png')

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            f.write("MAPE : %f\n\n" % mape(y_test, y_pred))
            print("MAPE ", mape(y_test, y_pred))
            f.close()

        if self.options['--multistep']:
            X_train, y_train, X_test, y_test, mean, std = load_data_days_more(path=self.options['-f'], shuffle=True)
            model_m = build_model_mlp(inputs=96*5, hiddens=200, outputs=96)
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            # train
            my_log = LossHistory()
            # log = model_m.fit(X_train, y_train, nb_epoch=100, batch_size=1, validation_split=0.1, verbose=2)
            log = model_m.fit(X_train, y_train, nb_epoch=100, batch_size=1, validation_data=(X_val, y_val), verbose=2, callbacks=[my_log])

            print log.history
            plt.figure(2)
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.savefig(self.options['-f'] + '_mlp_multistep_mse.png')

            # predict
            y_pred = model_m.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1],))

            f = open(self.options['-f'] + '_mlp_stats.txt', 'a')
            f.write("Multistep Stats\n")
            f.write("MAE : %f\n" % metrics.mean_absolute_error(y_test, y_pred))
            f.write("MSE : %f\n" % metrics.mean_squared_error(y_test, y_pred))

            print ("MAE ", metrics.mean_absolute_error(y_test, y_pred))
            print ("MSE ", metrics.mean_squared_error(y_test, y_pred))

            plt.figure(3, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            # plt.show()
            plt.savefig(self.options['-f'] + '_mlp_multistep_pred.png')

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            f.write("MAPE : %f\n" % mape(y_test, y_pred))
            print("MAPE ", mape(y_test, y_pred))
            f.close()

        if self.options['--onestep96']:
            X_train, y_train, X_test, y_test, mean, std = load_data_corr(path=self.options['-f'])
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            model = build_model_mlp(inputs=6, hiddens=5, outputs=1)
            # train
            # log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_split=0.1, verbose=2)
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_data=(X_val, y_val), verbose=2)
            print log.history
            plt.figure(4)
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            plt.show()
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0]))

            print("MAE  ", metrics.mean_absolute_error(y_test, y_pred))
            print("MSE  ", metrics.mean_squared_error(y_test, y_pred))

            plt.figure(5, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)

            y_test *= std
            y_test += mean
            y_pred *= std
            y_pred += mean
            print("MAPE ", mape(y_test, y_pred))


# files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
# for f in files:
#     run()

#
# def build_model_ffnn_add():
#     model = Sequential()
#     model.add(Dense(
#         output_dim=90,
#         input_shape=(96 * 2,),
#         activation='tanh'
#     ))
#
#     model.add(Dense(
#         output_dim=96,
#         activation='linear'
#     ))
#     start = time.time()
#     print ("Compilin")
#     model.compile(optimizer='SGD', loss='mse')
#     print ("Compiled and took ", time.time() - start, "seconds")
#     return model
#
#
# def run_add():
#     # X_train, y_train, X_test, y_test = load_data(sequence_length=96)
#     X_train, y_train, X_test, y_test = load_data_days_add()
#     model = build_model_ffnn_add()
#     model.fit(X=X_train, y=y_train, nb_epoch=20, batch_size=1, validation_split=0.1)
#     y_pred = model.predict(X_test)
#     y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
#     y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1],))
#     #
#     mape_val = mape(y_test, y_pred)
#     print (mapke(ypred=y_pred, ytrue=y_test))
#     print mape_val
#
#     print("MAE  ", metrics.mean_absolute_error(y_test, y_pred))
#     print("MSE  ", metrics.mean_squared_error(y_test, y_pred))
#     #
#     #
#     plt.style.use('ggplot')
#     plt.figure(0, figsize=(20, 4))
#     plt.plot(y_test)
#     plt.plot(y_pred)
#     plt.show()
#     return mape_val

# plt.style.use('ggplot')
# list1 = []
# list2 = []
# for dummy in xrange(20):
#     list1.append(run())
#     list2.append(run_add())
#
# print (list1)
# print (list2)
# # run()

# run_add()

# p = model.predict(X_test)
# y = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], ))
# pr = np.reshape(p, (p.shape[0]*p.shape[1], ))
# plt.figure()
# plt.plot(y)
# plt.plot(pr)
