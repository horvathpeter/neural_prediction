from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neupre.instructions.neuralops import mape

from neupre.misc.dataops import load_data_days_online
from neupre.misc.dataops import load_data_online
from neupre.misc.dataops import load_data_corr_online
from neupre.misc.dataops import load_data_point_online

plt.style.use('ggplot')


class BaseBackend(object):
    def __init__(self):

        # self.X_train_onestep = None
        # self.y_train_onestep = None

        self.X_train_multistep = None
        self.y_train_multistep = None

        # self.X_train_onestep96 = None
        # self.y_train_onestep96 = None

        self.X_train_start_date = None
        self.y_train_start_date = None

        self.last_line = 0
        # self.predictions_onestep = None
        self.predictions_multistep = None
        self.tensor3D = False
        self.path = None
        self.data_mean = None
        self.data_std = None

        self.maes = []
        self.mses = []
        self.mapes = []

        with open('mlp_online_stats.txt', 'w'):
            pass

    def sim(self):
        print("Skipujem ", self.last_line, " riadkov")
        new_data = load_data_point_online(maxline=96, skip=self.last_line, path=self.path)
        self.last_line += 96
        print("Budem skipovat ", self.last_line, " riadkov")

        # print("Onestep MAE was", mean_absolute_error(new_data[0], self.predictions_onestep[-1])) # first value was prediction
        # print("Onestep MSE was", mean_squared_error(new_data[0], self.predictions_onestep[-1]))

        print("Multistep MAE was ", mean_absolute_error(new_data, self.predictions_multistep[-1]))
        print("Multistep MSE was ", mean_squared_error(new_data, self.predictions_multistep[-1]))

        self.maes.append(mean_absolute_error(new_data, self.predictions_multistep[-1]))
        self.mses.append(mean_squared_error(new_data, self.predictions_multistep[-1]))

        f = open('mlp_online_stats.txt', 'a')
        f.write("Predikcia na %s\n" % (self.X_train_start_date + pd.DateOffset(days=70)).isoformat())
        f.write("MAE : %f\n" % mean_absolute_error(new_data, self.predictions_multistep[-1]))
        f.write("MSE : %f\n" % mean_squared_error(new_data, self.predictions_multistep[-1]))

        mean = 119694.44453877833
        std = 27297.4321577972
        y_test = np.copy(new_data)
        y_pred = np.copy(self.predictions_multistep[-1])

        y_test *= std
        y_test += mean
        y_pred *= std
        y_pred += mean
        f.write("MAPE : %f\n\n" % mape(y_test, y_pred))

        self.mapes.append(mape(y_test, y_pred))

        # print("New data is ", new_data)
        # print("X_train je ", self.X_train)

        # self.X_train_onestep = np.delete(self.X_train_onestep, range(96), 0)
        self.X_train_multistep = np.delete(self.X_train_multistep, 0, 0)

        if self.tensor3D:
            # self.X_train_multistep = np.vstack((self.X_train_multistep, np.reshape(self.y_train_multistep[-1], (
            #     1, self.y_train_multistep[-1].shape[0], 1))))
            new_entry = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]
            new_entry = np.reshape(new_entry, (1, new_entry.shape[0], 1))
            self.X_train_multistep = np.vstack((self.X_train_multistep, new_entry))

        else:
            # self.X_train_multistep = np.vstack((self.X_train_multistep, self.y_train_multistep[-1]))
            new_entry = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]
            self.X_train_multistep = np.vstack((self.X_train_multistep, new_entry))

        self.y_train_multistep = np.delete(self.y_train_multistep, 0, 0)
        self.y_train_multistep = np.vstack((self.y_train_multistep, new_data))
        # print("y_train po zmene je ", self.y_train)

        self.X_train_start_date += pd.DateOffset()
        self.y_train_start_date += pd.DateOffset()

        self.train()

        # X_test_multistep = self.y_train_multistep[-1]
        X_test_multistep = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]
        if self.tensor3D:
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0], 1))
        else:
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0]))
        # print("X_test_multistep je ", X_test_multistep)
        # p = self.model.predict(X_test_multistep)
        p = self.predict(X_test_multistep)
        self.predictions_multistep = np.vstack((self.predictions_multistep, p))
        # print("imp.predicitions je ", imp.predictions)
        plt.clf()
        self.plot()

    def plot(self):
        # p1 = self.predictions_onestep[-1]
        p2 = self.predictions_multistep[-1]

        train_data = self.X_train_multistep[:-1, :96]
        train_data = np.reshape(train_data, (train_data.shape[0] * train_data.shape[1]))
        train_data = np.append(train_data, self.X_train_multistep[-1])
        train_data = np.append(train_data, self.y_train_multistep[-1])

        train_index = pd.date_range(start=self.X_train_start_date, periods=70*96, freq='15T')
        p2_index = pd.date_range(start=train_index.date[-1]+pd.DateOffset(), periods=96, freq='15T')

        train_series = pd.Series(data=train_data, index=train_index)
        pred_series = pd.Series(data=p2, index=p2_index)

        plt.plot(train_series, color='blue')
        plt.plot(pred_series, color='green')

        try:
            p2before = self.predictions_multistep[-2]
            p2before_index = pd.date_range(start=train_index.date[-1], periods=96, freq='15T')
            p2before_series = pd.Series(data=p2before, index=p2before_index)
            plt.plot(p2before_series, color='red')
        except IndexError:
            print ("No real data yet")
            pass

        plt.draw()
        plt.pause(0.0001)
        #
        #
        #
        #
        # x = np.reshape(self.X_train_multistep, (self.X_train_multistep.shape[0] * self.X_train_multistep.shape[1]))
        # y = np.reshape(self.y_train_multistep, (self.y_train_multistep.shape[0] * self.y_train_multistep.shape[1]))
        #
        # index_x = pd.date_range(start=self.X_train_start_date, periods=69 * 96, freq='15T')
        # index_y = pd.date_range(start=self.y_train_start_date, periods=69 * 96, freq='15T')
        # index_p = pd.date_range(start=index_y.date[-1] + pd.DateOffset(), periods=96, freq='15T')
        #
        # xx = pd.Series(data=x, index=index_x)
        # yy = pd.Series(data=y, index=index_y)
        # pp = pd.Series(data=p2, index=index_p)
        #
        # plt.plot(xx, color='blue')
        # plt.plot(yy, color='blue')
        # plt.plot(pp, color='green')
        #
        # try:
        #     p_before = self.predictions_multistep[-2]
        #     index_p_before = pd.date_range(start=index_y.date[-1], periods=96, freq='15T')
        #     pp_before = pd.Series(data=p_before, index=index_p_before)
        #     plt.plot(pp_before, color='red')
        # except IndexError:
        #     print ("No real data yet")
        #     pass
        #
        # plt.draw()
        # plt.pause(0.0001)

    def initialize(self, is_rec, path):
        self.tensor3D = is_rec
        self.path = path

        # self.X_train_onestep, self.y_train_onestep = load_data_online(maxline=96 * 70, reshape=self.tensor3D)
        self.X_train_multistep, self.y_train_multistep = load_data_days_online(maxline=96 * 70, reshape=self.tensor3D,
                                                                               path=self.path)
        # self.X_train_onestep96, self.y_train_onestep96 = load_data_corr_online(maxline=96*70, reshape=self.tensor3D)
        self.last_line = 96 * 70

        # print("Budem skipovat ", self.last_line, " riadkov")
        self.X_train_start_date = pd.Timestamp('2013-07-01', offset='D')
        self.y_train_start_date = pd.Timestamp('2013-07-02', offset='D')

        # initial training
        log1 = self.train()

        # X_test_onestep = np.delete(np.append(self.X_train_onestep[-1], self.y_train_onestep[-1]), 0)

        # X_test_multistep = self.y_train_multistep[-1]
        X_test_multistep = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]

        if self.tensor3D:
            # X_test_onestep = np.reshape(X_test_onestep, (1, X_test_onestep.shape[0], 1))
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0], 1))
        else:
            # X_test_onestep = np.reshape(X_test_onestep, (1, X_test_onestep.shape[0]))
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0]))

        # print("X_test_onestep je ", X_test_onestep)
        # print("X_test_multistep je ", X_test_multistep)

        p2 = self.predict(X_test_multistep)
        # self.predictions_onestep = np.array(p1)
        self.predictions_multistep = np.array(p2)
        # print("imp.predicitions je ", self.predictions_onestep)
        # print("imp.predicitions je ", self.predictions_multistep)
        plt.ion()
        plt.figure(0, figsize=(20, 4))
        self.plot()
        plt.show()

    def train(self):
        raise NotImplementedError("The train() method is not implemented!")

    def predict(self, X_test2):
        raise NotImplementedError("The predict() method is not implemented!")
