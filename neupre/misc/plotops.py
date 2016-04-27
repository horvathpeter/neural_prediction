import matplotlib
import matplotlib.pyplot as plt

import pandas as pd


class Plotter(object):
    def __init__(self):
        matplotlib.style.use('ggplot')

        # plt.figure(0)
        # self.ax1 = fig_all.add_subplot(111)
        # if plt.isinteractive() is False:
        #     plt.ion()

        # plt.figure(1)
        # self.ax2 = fig_mapky.add_subplot(111)
        # if plt.isinteractive() is False:
        #     plt.ion()

    def clean(self):
        plt.figure(0)
        plt.clf()
        plt.figure(1)
        plt.clf()

    def plot_to_see(self, training_set, new_data, prediction):
        plt.figure(0)
        pd.concat(training_set.values()).plot(title="Neural Prediction")
        new_data.plot(color="red")
        prediction.plot(color="green")
        #
        # self.ax1.plot(pd.concat(training_set.values()))
        # self.ax1.plot(new_data)
        # self.ax1.plot(prediction)

        plt.legend(['training', 'real', 'predicted'], loc='best')
        plt.draw()

    def plot_list(self, listik):
        plt.figure(1)
        plt.title("MAPE")
        plt.plot(listik)

if __name__ == "__main__":
    print ("Initialized plotops.py")
