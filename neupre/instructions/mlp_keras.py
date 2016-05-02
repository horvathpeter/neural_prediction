import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn import metrics


class ImplementationFFNN(object):
    def __init__(self, input_dim, hidden_dim_1, output_dim, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.X_train = None
        self.y_train = None

        self.X_train_start_date = None
        self.y_train_start_date = None

        self.last_line = 0
        self.predictions = None

        self.model = Sequential()
        self.model.add(Dense(
            output_dim=hidden_dim_1,
            input_shape=(input_dim,),
            activation='tanh'
        ))

        self.model.add(Dense(
            output_dim=output_dim,
            activation='linear'
        ))
        start = time.time()
        print ("Compilin")
        self.model.compile(optimizer='SGD', loss='mse')
        print ("Compiled and took ", time.time() - start, "seconds")

    def train(self):
        log = self.model.fit(self.X_train, self.y_train, batch_size=1, nb_epoch=10, validation_split=0.1, verbose=0)
        return log

    def predict(self, X_test):
        return self.model.predict(X_test)

    def sim(self):
        print("Skipujem ", self.last_line, " riadkov ")
        new_data = load_data_point(96, self.last_line)
        self.last_line += 96
        print("Budem skipovat ", self.last_line, " riadkov")

        print("MAE was ", metrics.mean_absolute_error(new_data, self.predictions[-1]))
        print("MSE was ", metrics.mean_squared_error(new_data, self.predictions[-1]))
        # print("New misc is ", new_data)
        # print("X_train je ", self.X_train)
        self.X_train = np.delete(self.X_train, 0, 0)
        self.X_train = np.vstack((self.X_train, self.y_train[-1]))
        # print("X_train po zmene je ", self.X_train)

        # print("y_train je ", self.y_train)
        self.y_train = np.delete(self.y_train, 0, 0)
        self.y_train = np.vstack((self.y_train, new_data))
        # print("y_train po zmene je ", self.y_train)

        self.X_train_start_date += pd.DateOffset()
        self.y_train_start_date += pd.DateOffset()

        self.train()
        X_test = self.y_train[-1]
        X_test = np.reshape(X_test, (1, X_test.shape[0]))
        # print("X_test je ", X_test)
        p = self.model.predict(X_test)
        self.predictions = np.vstack((self.predictions, p))
        # print("imp.predicitions je ", imp.predictions)
        plt.clf()
        self.plot()

    def plot(self):
        p = self.predictions[-1]

        x = np.reshape(self.X_train, (self.X_train.shape[0] * self.X_train.shape[1]))
        y = np.reshape(self.y_train, (self.y_train.shape[0] * self.y_train.shape[1]))

        index_x = pd.date_range(start=self.X_train_start_date, periods=69 * 96, freq='15T')
        index_y = pd.date_range(start=self.y_train_start_date, periods=69 * 96, freq='15T')
        index_p = pd.date_range(start=index_y.date[-1] + pd.DateOffset(), periods=96, freq='15T')

        xx = pd.Series(data=x, index=index_x)
        yy = pd.Series(data=y, index=index_y)
        pp = pd.Series(data=p, index=index_p)

        # print("XX plot ", xx
        # print("YY plot ", yy
        # print("pp plot ", pp

        plt.plot(xx, color='blue')
        plt.plot(yy, color='blue')
        plt.plot(pp, color='green')

        try:
            p_before = self.predictions[-2]
            index_p_before = pd.date_range(start=index_y.date[-1], periods=96, freq='15T')
            pp_before = pd.Series(data=p_before, index=index_p_before)
            plt.plot(pp_before, color='red')
        except IndexError:
            print ("No real misc yet")
            pass

        plt.draw()
        plt.pause(0.0001)
