from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neupre.instructions.base import mape
from neupre.misc.dataops import load_data_point_online
plt.style.use('ggplot')


class BaseBackend(object):
    def __init__(self, buffsize):

        # self.X_train_onestep = None
        # self.y_train_onestep = None

        self.X_train_multistep = None
        self.y_train_multistep = None

        self.X_train_onestep96 = None
        self.y_train_onestep96 = None
        self.X_test_onestep96 = None

        self.X_train_start_date = None
        self.y_train_start_date = None

        self.counter = 0
        self.last_line = 0
        # self.predictions_onestep = None
        self.predictions_multistep = None
        self.predictions_onestep96 = None
        self.tensor3D = False
        self.path = None
        self.data_mean = None
        self.data_std = None
        self.statspath = None
        self.buffsize = buffsize

        self.maes_multi = []
        self.mses_multi = []
        self.mapes_multi = []

        self.maes_one96 = []
        self.mses_one96 = []
        self.mapes_one96 = []

        self.mapes_working_days_multi = []
        self.mapes_holidays_multi = []
        self.mapes_working_days_one96 = []
        self.mapes_holidays_one96 = []

        self.mapes_jan_one96 = []
        self.mapes_feb_one96 = []
        self.mapes_mar_one96 = []
        self.mapes_apr_one96 = []
        self.mapes_may_one96 = []
        self.mapes_jun_one96 = []
        self.mapes_jul_one96 = []
        self.mapes_aug_one96 = []
        self.mapes_sep_one96 = []
        self.mapes_oct_one96 = []
        self.mapes_nov_one96 = []
        self.mapes_dec_one96 = []

        self.mapes_jan_multi = []
        self.mapes_feb_multi = []
        self.mapes_mar_multi = []
        self.mapes_apr_multi = []
        self.mapes_may_multi = []
        self.mapes_jun_multi = []
        self.mapes_jul_multi = []
        self.mapes_aug_multi = []
        self.mapes_sep_multi = []
        self.mapes_oct_multi = []
        self.mapes_nov_multi = []
        self.mapes_dec_multi = []



    def sim(self):
        print("Skipping ", self.last_line, " lines")
        new_data = load_data_point_online(maxline=96, skip=self.last_line, path=self.path)
        self.last_line += 96
        print("Will skip ", self.last_line, " lines")

        # print("Onestep MAE was", mean_absolute_error(new_data[0], self.predictions_onestep[-1])) # first value was prediction
        # print("Onestep MSE was", mean_squared_error(new_data[0], self.predictions_onestep[-1]))

        # calculate errors of predictions in last step

        self.maes_multi.append(mean_absolute_error(new_data, self.predictions_multistep[-1]))
        self.mses_multi.append(mean_squared_error(new_data, self.predictions_multistep[-1]))

        self.maes_one96.append(mean_absolute_error(new_data, self.predictions_onestep96[-1]))
        self.mses_one96.append(mean_squared_error(new_data, self.predictions_onestep96[-1]))

        y_test = np.copy(new_data)
        y_pred_multi = np.copy(self.predictions_multistep[-1])
        y_pred_one96 = np.copy(self.predictions_onestep96[-1])

        y_test *= self.data_std
        y_test += self.data_mean
        y_pred_multi *= self.data_std
        y_pred_multi += self.data_mean

        y_pred_one96 *= self.data_std
        y_pred_one96 += self.data_mean

        self.mapes_multi.append(mape(y_test, y_pred_multi))
        self.mapes_one96.append(mape(y_test, y_pred_one96))

        prediction_day = self.X_train_start_date + pd.DateOffset(days=self.buffsize)

        # statistics...
        if prediction_day.dayofweek >= 5:
            self.mapes_holidays_one96.append(self.mapes_one96[-1])
            self.mapes_holidays_multi.append(self.mapes_multi[-1])
        elif prediction_day.dayofweek < 5 and prediction_day not in pd.date_range(start='2013-12-21', periods=24)\
                and prediction_day not in pd.date_range(start='2014-04-15', periods=20):
            self.mapes_working_days_one96.append(self.mapes_one96[-1])
            self.mapes_working_days_multi.append(self.mapes_multi[-1])

        if prediction_day.month == 1:
            self.mapes_jan_one96.append(self.mapes_one96[-1])
            self.mapes_jan_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 2:
            self.mapes_feb_one96.append(self.mapes_one96[-1])
            self.mapes_feb_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 3:
            self.mapes_mar_one96.append(self.mapes_one96[-1])
            self.mapes_mar_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 4:
            self.mapes_apr_one96.append(self.mapes_one96[-1])
            self.mapes_apr_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 5:
            self.mapes_may_one96.append(self.mapes_one96[-1])
            self.mapes_may_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 6:
            self.mapes_jun_one96.append(self.mapes_one96[-1])
            self.mapes_jun_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 7:
            self.mapes_jul_one96.append(self.mapes_one96[-1])
            self.mapes_jul_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 8:
            self.mapes_aug_one96.append(self.mapes_one96[-1])
            self.mapes_aug_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 9:
            self.mapes_sep_one96.append(self.mapes_one96[-1])
            self.mapes_sep_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 10:
            self.mapes_oct_one96.append(self.mapes_one96[-1])
            self.mapes_oct_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 11:
            self.mapes_nov_one96.append(self.mapes_one96[-1])
            self.mapes_nov_multi.append(self.mapes_multi[-1])
        elif prediction_day.month == 12:
            self.mapes_dec_one96.append(self.mapes_one96[-1])
            self.mapes_dec_multi.append(self.mapes_multi[-1])

        print("Prediction for %s\n" % prediction_day.isoformat())
        print("Multistep MAE was ", self.maes_multi[-1])
        print("Multistep MSE was ", self.mses_multi[-1])
        print("Multistep MAPE was ", self.mapes_multi[-1])

        print("Onestep96 MAE was ", self.maes_one96[-1])
        print("Onestep96 MSE was ", self.mses_one96[-1])
        print("Onestep96 MAPE was ", self.mapes_one96[-1])

        with open('%s/stats.txt' % self.statspath, 'a') as f:
            f.write("%d\n" % self.counter)
            f.write("Prediction for %s\n" % prediction_day.isoformat())
            f.write("MAE multi: %f\n" % self.maes_multi[-1])
            f.write("MSE multi: %f\n" % self.mses_multi[-1])
            f.write("MAPE multi: %f\n" % self.mapes_multi[-1])

            f.write("MAE one96: %f\n" % self.maes_one96[-1])
            f.write("MSE one96: %f\n" % self.mses_one96[-1])
            f.write("MAPE one96: %f\n\n" % self.mapes_one96[-1])

        self.counter += 1

        # create new training set, inputs and predict the next step

        # self.X_train_onestep = np.delete(self.X_train_onestep, range(96), 0)
        self.X_train_multistep = np.delete(self.X_train_multistep, 0, 0)
        self.X_train_onestep96 = np.delete(self.X_train_onestep96, range(96), 0)

        if self.tensor3D:
            self.X_test_onestep96 = np.reshape(self.X_test_onestep96, (self.X_test_onestep96.shape[0], self.X_test_onestep96.shape[1], 1))
        self.X_train_onestep96 = np.append(self.X_train_onestep96, self.X_test_onestep96, 0)

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

        self.y_train_onestep96 = np.delete(self.y_train_onestep96, range(96), 0)
        self.y_train_onestep96 = np.append(self.y_train_onestep96, new_data)

        self.X_train_start_date += pd.DateOffset()
        self.y_train_start_date += pd.DateOffset()

        self.train()

        self.X_test_onestep96 = []
        for index in reversed(range(96)):
            index += 1
            self.X_test_onestep96.append([
                self.y_train_onestep96[-index - 96 * 13],
                self.y_train_onestep96[-index - 96 * 7],
                self.y_train_onestep96[-index - 96 * 6 - 1],
                self.y_train_onestep96[-index - 96 * 6],
                self.y_train_onestep96[-index - 96 * 5],
                self.y_train_onestep96[-index]
            ])
        self.X_test_onestep96 = np.array(self.X_test_onestep96)

        # X_test_multistep = self.y_train_multistep[-1]
        X_test_multistep = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]
        if self.tensor3D:
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0], 1))
            self.X_test_onestep96 = np.reshape(self.X_test_onestep96,
                                               (self.X_test_onestep96.shape[0], self.X_test_onestep96.shape[1], 1))
        else:
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0]))

        p2, p3 = self.predict(X_test_multistep, self.X_test_onestep96)
        p3 = np.reshape(p3, (1, p3.shape[0]))
        self.predictions_multistep = np.vstack((self.predictions_multistep, p2))
        self.predictions_onestep96 = np.vstack((self.predictions_onestep96, p3))

        # TODO try without clf, because of draw
        plt.clf()
        self.plot()

    def plot(self):
        # p1 = self.predictions_onestep[-1]
        p2 = self.predictions_multistep[-1]
        p3 = self.predictions_onestep96[-1]

        train_data = self.X_train_multistep[:-1, :96]
        train_data = np.reshape(train_data, (train_data.shape[0] * train_data.shape[1]))
        train_data = np.append(train_data, self.X_train_multistep[-1])
        train_data = np.append(train_data, self.y_train_multistep[-1])

        train_index = pd.date_range(start=self.X_train_start_date, periods=self.buffsize * 96, freq='15T')
        p3_index = p2_index = pd.date_range(start=train_index.date[-1] + pd.DateOffset(), periods=96, freq='15T')

        train_series = pd.Series(data=train_data, index=train_index)
        pred2_series = pd.Series(data=p2, index=p2_index)
        pred3_series = pd.Series(data=p3, index=p3_index)

        plt.plot(train_series, color='blue')
        plt.plot(pred2_series, color='#00cc00')
        plt.plot(pred3_series, color='#ff0000')

        try:
            p2before = self.predictions_multistep[-2]
            p3before = self.predictions_onestep96[-2]

            p3before_index = p2before_index = pd.date_range(start=train_index.date[-1], periods=96, freq='15T')
            p2before_series = pd.Series(data=p2before, index=p2before_index)
            p3before_series = pd.Series(data=p3before, index=p3before_index)
            plt.plot(p2before_series, color='#006600')
            plt.plot(p3before_series, color='#cc0000')

        except IndexError:
            print ("No real data yet")
            pass

        plt.title('Online prediction of %s' % self.path)
        plt.ylabel('Consumption (z-score)')
        plt.legend(['training', 'MultiStep', 'Onestep96', 'MultiStep\naccuracy', 'Onestep96\naccuracy'], loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True)

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

    def initialize(self, is_rec, path, mean, std, statspath):
        self.tensor3D = is_rec
        self.path = path
        self.data_mean = mean
        self.data_std = std
        self.statspath = statspath

        from neupre.misc.dataops import load_data_days_online
        from neupre.misc.dataops import load_data_corr_online

        with open('%s/stats.txt' % self.statspath, 'w+'):
            pass

        # self.X_train_onestep, self.y_train_onestep = load_data_online(maxline=96 * self.buffsize, reshape=self.tensor3D)
        self.X_train_multistep, self.y_train_multistep = load_data_days_online(maxline=96 * self.buffsize,
                                                                               reshape=self.tensor3D,
                                                                               path=self.path)

        self.X_train_onestep96, self.y_train_onestep96 = load_data_corr_online(maxline=96 * self.buffsize,
                                                                               reshape=self.tensor3D,
                                                                               path=self.path)
        self.last_line = 96 * self.buffsize

        self.X_train_start_date = pd.Timestamp('2013-07-01', offset='D')
        self.y_train_start_date = pd.Timestamp('2013-07-02', offset='D')

        # initial training
        log2, log3 = self.train()

        # for prediction 15 mins ahead
        # X_test_onestep = np.delete(np.append(self.X_train_onestep[-1], self.y_train_onestep[-1]), 0)

        # when training X has just one day, instead of 5
        # X_test_multistep = self.y_train_multistep[-1]

        # next day forecast input (1 output neuron)
        self.X_test_onestep96 = []
        for index in reversed(range(96)):
            index += 1
            self.X_test_onestep96.append([
                self.y_train_onestep96[-index - 96 * 13],
                self.y_train_onestep96[-index - 96 * 7],
                self.y_train_onestep96[-index - 96 * 6 - 1],
                self.y_train_onestep96[-index - 96 * 6],
                self.y_train_onestep96[-index - 96 * 5],
                self.y_train_onestep96[-index]
            ])
        self.X_test_onestep96 = np.array(self.X_test_onestep96)

        # next day forecast input (96 output neuronos)
        X_test_multistep = np.append(self.X_train_multistep[-1], self.y_train_multistep[-1])[96:]

        if self.tensor3D:
            # X_test_onestep = np.reshape(X_test_onestep, (1, X_test_onestep.shape[0], 1))
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0], 1))
            self.X_test_onestep96 = np.reshape(self.X_test_onestep96, (self.X_test_onestep96.shape[0], self.X_test_onestep96.shape[1], 1))
        else:
            # X_test_onestep = np.reshape(X_test_onestep, (1, X_test_onestep.shape[0]))
            X_test_multistep = np.reshape(X_test_multistep, (1, X_test_multistep.shape[0]))

        # print("X_test_onestep ===> ", X_test_onestep)
        # print("X_test_multistep ===> ", X_test_multistep)

        p2, p3 = self.predict(X_test_multistep, self.X_test_onestep96)
        # print p2
        # print p3
        p3 = np.reshape(p3, (1, p3.shape[0]))
        # print p3
        # self.predictions_onestep = np.array(p1)
        self.predictions_multistep = np.array(p2)
        self.predictions_onestep96 = np.array(p3)
        plt.ion()
        plt.figure(0, figsize=(17, 4))
        plt.title('Online prediction of %s' % self.path)
        plt.ylabel('Consumption (z-score)')
        self.plot()
        plt.show()

    def train(self):
        raise NotImplementedError("The train() method is not implemented!")

    def predict(self, X_test2, Xtest3):
        raise NotImplementedError("The predict() method is not implemented!")
