import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neupre.misc.dataops import load_data_point_online
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Mlp(object):
    def __init__(self, input_dim, hidden_dim_1, output_dim, learning_rate):
        from neupre.misc.builders import build_model_mlp

        self.model = build_model_mlp(input_dim, hidden_dim_1, output_dim)

        self.X_train = None
        self.y_train = None

        self.X_train_start_date = None
        self.y_train_start_date = None

        self.last_line = 0
        self.predictions = None

    def train(self):
        log = self.model.fit(self.X_train, self.y_train, batch_size=1, nb_epoch=20, validation_split=0.1, verbose=2)
        return log

    def predict(self, X_test):
        return self.model.predict(X_test)

    def sim(self):
        print("Skipujem ", self.last_line, " riadkov ")
        new_data = load_data_point_online(maxline=96, skip=self.last_line)
        self.last_line += 96
        print("Budem skipovat ", self.last_line, " riadkov")

        print("MAE was ", mean_absolute_error(new_data, self.predictions[-1]))
        print("MSE was ", mean_squared_error(new_data, self.predictions[-1]))
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

        plt.plot(xx, color='blue')
        plt.plot(yy, color='blue')
        plt.plot(pp, color='green')

        try:
            p_before = self.predictions[-2]
            index_p_before = pd.date_range(start=index_y.date[-1], periods=96, freq='15T')
            pp_before = pd.Series(data=p_before, index=index_p_before)
            plt.plot(pp_before, color='red')
        except IndexError:
            print ("No real data yet")
            pass

        plt.draw()
        plt.pause(0.0001)
