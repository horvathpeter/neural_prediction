from collections import OrderedDict

import numpy as np
import pandas as pd

from pybrain.datasets import UnsupervisedDataSet, SupervisedDataSet
from pybrain.structure import LinearLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mape(real, pred):
    """Returns mean average percentage error"""
    return 100 * np.mean(np.abs(real - pred) / real)


def mapke(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100 * np.mean(np.abs(ypred[idx] - ytrue[idx]) / ytrue[idx])


# TODO instructions class for other models - interface
class Abstract(object):
    def __init__(self):
        pass

    def train(self, rng):
        raise NotImplementedError


class ImplementationFFNN(object):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # PyBrain feedforward neural network is a instructions of this model implementation
        self.net = buildNetwork(self.input_dim, self.hidden_dim_1, self.hidden_dim_2, self.output_dim,
                                hiddenclass=TanhLayer, outclass=LinearLayer, bias=True, recurrent=False)
        self.training_set = OrderedDict()
        self.last_day = None
        self.epochs = 100
        self.plotter = Plotter()

    def train(self, rng):
        mapky = []
        super_ds = SupervisedDataSet(672, 96)
        for actual_day in rng[:-1]:
            day_after = actual_day + pd.DateOffset(days=1)
            # print "Getting misc for actual day ", actual_day, " and for next day ", day_after

            sample_input = self.training_set[actual_day.isoformat()]
            sample_target = self.training_set[day_after.isoformat()][6::7]

            super_ds.addSample(sample_input, sample_target)
        backprop_tr = BackpropTrainer(self.net, dataset=super_ds)
        backprop_tr.trainEpochs(100)
        # self.net.resetDerivatives()
        # out = self.net.activate(sample_input)
        # err = sample_target - out
        # self.net.backActivate(err)
        # self.net._setParameters(self.net.params + self.learning_rate * self.net.derivs)

        # mapka = mape(time_series_real=sample_target, time_series_predicted=out)
        # # print "From", actual_day, "to", day_after, "Error is ", mapka
        # mapky.append(mapka)

        # return mapky
        return reduce(lambda x, y: x + y, mapky) / len(mapky)

    def trainEpochs(self, epochs, *args):
        if isinstance(args[0], pd.DatetimeIndex):
            avg_mapky = []
            print("Going to train neural network on set ", args[0][0], " to ", args[0][-1], " ", epochs, " times")
            for dummy in range(epochs):
                avg_mapky.append(self.train(*args))
                # avg_mapka = self.train(*args)
                # print "Average mapka is ", avg_mapka
                # avg_mapky.append(avg_mapka)
                # self.plotter.plot_list(avg_mapky)
        return avg_mapky

    def predict(self, network_input, forecast_date):
        """Activates the neural network on the given input and returns the output
        :param network_input:
        :param forecast_date:
        :return:
        """
        index_range = pd.date_range(start=forecast_date, periods=96, freq='15T')
        activation_set = UnsupervisedDataSet(96, )
        activation_set.addSample(network_input)
        activation = self.net.activateOnDataset(activation_set)[0]
        return pd.Series(data=activation, index=index_range)

    def data_arrival(self, new_data, data_day):
        """Simulation of new data point arrival - training dataset modificaiton first day of the dataset goes out
        and the new data point is appended to the dataset, so the sliding window size is the same - the
        neural network is is trained on this new window each sample is backpropagated through the network only once
        :param new_data:
        :param data_day: the date of the new measurements
        :return:
        """
        print("Going to modify the training set ")

        self.training_set.popitem(last=False)
        self.training_set[data_day.isoformat()] = new_data

        # print "Dataset modified now has keys ", self.training_set.keys()
        new_range = pd.date_range(self.training_set.keys()[0], periods=70)

        # re-train the model in the instructions of the new misc
        self.trainEpochs(self.epochs, new_range)

    def simulate_next_step(self):
        self.last_day += pd.DateOffset(days=1)

        new_data = data_handler.load_data_point(self.last_day)
        print("Mape of the prediction for ", self.last_day, " is ", mape(new_data,
                                                                         predictions[self.last_day.isoformat()]))
        self.plotter.clean()
        self.plotter.plot_to_see(self.training_set, new_data, predictions[self.last_day.isoformat()])

        # retrain the model with new misc set
        self.data_arrival(new_data, self.last_day)

        day_after = self.last_day + pd.DateOffset()
        print("Saving predictions for ", day_after)
        predictions[day_after.isoformat()] = self.predict(new_data, day_after)


class ImplementationLSTM(object):
    # TODO implement model using recurrent LSTM network
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


class StateModel(object):
    def __init__(self, implementation):
        self.__implementation = implementation

    def change_implementation(self, implementation):
        self.__implementation = implementation

    def __getattr__(self, item):
        return getattr(self.__implementation, item)


if __name__ == "__main__":
    model = ImplementationFFNN(672, 336, 168, 96, 0.01)

    predictions = OrderedDict()
    data_handler = DataHandler('91_trnava_suma.csv')

    initial_range = pd.date_range('2013-07-01', periods=70)
    model.training_set = data_handler.load_training_set(initial_range)
    model.last_day = initial_range[-1]

    # train model on the initial data_set
    mapy = model.train(initial_range)

    # mapy = model.trainEpochs(model.epochs, initial_range)
    print(mapy)

    # next_day = initial_range[-1] + pd.DateOffset(days=1)
    # # predict the next day
    # predictions[next_day.isoformat()] = model.predict(model.training_set.values()[-1], next_day)
    # print "Forecast for one day ahead for ", next_day, " is ", predictions[next_day.isoformat()].tolist()
