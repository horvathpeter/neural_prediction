import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .base import Base
plt.style.use('ggplot')


# def load_data_days(maxline=None):
#     path_to_dataset = 'neupre/misc/data/91_trnava_suma_stand.csv'
#     with open(path_to_dataset) as f:
#         data = csv.reader(f, delimiter=',')
#         power = []
#         nb_of_values = 0
#         for line in data:
#             try:
#                 power.append(float(line[2]))
#                 nb_of_values += 1
#             except ValueError:
#                 pass
#             if nb_of_values >= maxline:
#                 break
#     print ("Data loaded from csv. Formatting...")
#     result = []
#     for index in np.arange(start=0, stop=maxline - 96, step=96):
#         result.append(power[index: index + 96 * 2])
#
#     result = np.array(result)
#     # result_mean = result.mean()
#     # result_std = result.std()
#     # result -= result_mean
#     # result /= result_std
#     # # result = zscore(result)
#
#     print ("Data shape : ", result.shape)
#
#     half = result.shape[1] / 2
#     X_train = result[:, :half]
#     y_train = result[:, half:]
#
#     return [X_train, y_train]


# def load_data_point(maxline=96, skip=None):
#     path_to_dataset = 'neupre/misc/data/91_trnava_suma_stand.csv'
#     with open(path_to_dataset) as f:
#         data = pd.read_csv(f, skiprows=skip, header=None, nrows=96)
#     print ("Data loaded from csv. Formatting...")
#     result = np.array(data[2])
#     # result_mean = result.mean()
#     # result_std = result.std()
#     # result -= result_mean
#     # result /= result_std
#
#     print ("Data shape : ", result.shape)
#     return result


class OnlineMlp(Base):
    def run(self):
        from neupre.misc.dataops import load_data_days_online
        from neupre.backend.onlinemlp_backend import Mlp

        imp = Mlp(96, 50, 96, 0.01)

        imp.X_train, imp.y_train = load_data_days_online(maxline=96 * 70)

        imp.last_line = 96 * 70
        print("Budem skipovat ", imp.last_line, " riadkov")
        imp.X_train_start_date = pd.Timestamp('2013-07-01', offset='D')
        imp.y_train_start_date = pd.Timestamp('2013-07-02', offset='D')
        log = imp.train()

        X_test = imp.y_train[-1]
        X_test = np.reshape(X_test, (1, X_test.shape[0]))
        print("X_test je ", X_test)
        p = imp.predict(X_test)
        imp.predictions = np.array(p)
        print("imp.predicitions je ", imp.predictions)
        plt.ion()
        plt.figure(0, figsize=(20, 4))
        plt.show()
        imp.plot()

        # TODO simtoend
        if self.options['--simsteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()
        elif self.options['--simtoend']:
            pass


# if __name__ == "__main__":
#     imp = OnlineMlp(96, 50, 96, 0.01)
#
#     imp.X_train, imp.y_train = load_data_days_online(maxline=96 * 70)
#
#     imp.last_line = 96 * 70
#     print("Budem skipovat ", imp.last_line, " riadkov")
#     imp.X_train_start_date = pd.Timestamp('2013-07-01', offset='D')
#     imp.y_train_start_date = pd.Timestamp('2013-07-02', offset='D')
#     log = imp.train()
#
#     X_test = imp.y_train[-1]
#     X_test = np.reshape(X_test, (1, X_test.shape[0]))
#     print("X_test je ", X_test)
#     p = imp.predict(X_test)
#     imp.predictions = np.array(p)
#     print("imp.predicitions je ", imp.predictions)
#     plt.figure(0, figsize=(20, 4))
#     plt.show()
#     plt.ion()
#     imp.plot()
#     # imp.sim()
#     # # #
#     # for dummy in xrange(200):
#     #     imp.sim()
