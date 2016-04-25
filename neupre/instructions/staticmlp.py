import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import Base

plt.style.use('ggplot')


class StaticMlp(Base):
    def run(self):
        from sklearn import metrics
        from neupre.misc.dataops import load_data, load_data_days
        from neupre.misc.builders import build_model_mlp
        # print self.options
        if self.options['--onestep']:
            X_train, y_train, X_test, y_test = load_data(sequence_length=96)
            model = build_model_mlp(inputs=95, hiddens=50, outputs=1)
            # train
            model.fit(X_train, y_train, nb_epoch=100, batch_size=100, validation_split=0.1, verbose=2)
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))

            # mape_val = mape(y_test, y_pred)
            # print (mapke(ypred=y_pred, ytrue=y_test))
            # print mape_val

            print("MAE  ", metrics.mean_absolute_error(y_test, y_pred))
            print("MSE  ", metrics.mean_squared_error(y_test, y_pred))

            plt.figure(0, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.show()
        elif self.options['--multistep']:
            X_train, y_train, X_test, y_test = load_data_days()
            model = build_model_mlp(inputs=96, hiddens=50, outputs=96)
            # train
            model.fit(X_train, y_train, nb_epoch=20, batch_size=1, validation_split=0.1, verbose=2)
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1],))

            print ("MAE ", metrics.mean_absolute_error(y_test, y_pred))
            print ("MSE ", metrics.mean_squared_error(y_test, y_pred))

            plt.figure(0, figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.show()



def build_model_ffnn_add():
    model = Sequential()
    model.add(Dense(
        output_dim=90,
        input_shape=(96 * 2,),
        activation='tanh'
    ))

    model.add(Dense(
        output_dim=96,
        activation='linear'
    ))
    start = time.time()
    print ("Compilin")
    model.compile(optimizer='SGD', loss='mse')
    print ("Compiled and took ", time.time() - start, "seconds")
    return model


def run_add():
    # X_train, y_train, X_test, y_test = load_data(sequence_length=96)
    X_train, y_train, X_test, y_test = load_data_days_add()
    model = build_model_ffnn_add()
    model.fit(X=X_train, y=y_train, nb_epoch=20, batch_size=1, validation_split=0.1)
    y_pred = model.predict(X_test)
    y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
    y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1],))
    #
    mape_val = mape(y_test, y_pred)
    print (mapke(ypred=y_pred, ytrue=y_test))
    print mape_val

    print("MAE  ", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE  ", metrics.mean_squared_error(y_test, y_pred))
    #
    #
    plt.style.use('ggplot')
    plt.figure(0, figsize=(20, 4))
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()
    return mape_val

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
