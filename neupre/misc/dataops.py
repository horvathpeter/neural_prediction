import csv
import numpy as np
import random
import pandas as pd


def load_data(dataset=None,
              path_to_dataset='neupre/misc/data/91_trnava_suma_.csv',
              ratio=1.0,
              sequence_length=96,
              shuffle=False,
              zscore=True,
              reshape=False):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path_to_dataset:
        max_values = ratio * 57216
        with open(path_to_dataset) as f:
            data = csv.reader(f, delimiter=",")
            power = []
            nb_of_values = 0
            for line in data:
                try:
                    power.append(float(line[2]))
                    nb_of_values += 1
                except ValueError:
                    pass
                if nb_of_values >= max_values:
                    break
        print ("Data loaded from csv. Formatting...")

    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)

    if zscore:
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        print ("Shift : ", result_mean)

    print ("Data shape  : ", result.shape)

    row = int(round(0.95 * result.shape[0]))

    train = result[:row, :]
    if shuffle:
        np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]


def load_data_days(dataset=None,
                   path_to_dataset='neupre/misc/data/91_trnava_suma_.csv',
                   ratio=1.0,
                   sequence_length=96,
                   shuffle=False,
                   zscore=True,
                   reshape=False):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path_to_dataset:
        max_values = ratio * 57216
        with open(path_to_dataset) as f:
            data = csv.reader(f, delimiter=",")
            power = []
            nb_of_values = 0
            for line in data:
                try:
                    power.append(float(line[2]))
                    nb_of_values += 1
                except ValueError:
                    pass
                if nb_of_values >= max_values:
                    break
        print ("Data loaded from csv. Formatting...")

    result = []
    for index in np.arange(start=0, stop=57120, step=96):
        result.append(power[index: index + 96 * 2])

    result = np.array(result)
    if zscore:
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        print ("Shift : ", result_mean)

    print ("Data shape : ", result.shape)

    row = int(round(0.95 * result.shape[0]))

    train = result[:row, :]
    if shuffle:
        np.random.shuffle(train)
    half = train.shape[1] / 2
    X_train = train[:, :half]
    y_train = train[:, half:]
    X_test = result[row:, :half]
    y_test = result[row:, half:]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]


def load_data_days_online(maxline,
                          dataset=None,
                          path_to_dataset='neupre/misc/data/91_trnava_suma_stand.csv'):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path_to_dataset:
        with open(path_to_dataset) as f:
            data = csv.reader(f, delimiter=',')
            power = []
            nb_of_values = 0
            for line in data:
                try:
                    power.append(float(line[2]))
                    nb_of_values += 1
                except ValueError:
                    pass
                if nb_of_values >= maxline:
                    break
        print ("Data loaded from csv. Formatting...")

    result = []
    for index in np.arange(start=0, stop=maxline - 96, step=96):
        result.append(power[index: index + 96 * 2])

    result = np.array(result)
    print ("Data shape : ", result.shape)
    half = result.shape[1] / 2
    X_train = result[:, :half]
    y_train = result[:, half:]

    return [X_train, y_train]


def load_data_point_online(maxline=96,
                           path_to_dataset='neupre/misc/data/91_trnava_suma_stand.csv',
                           skip=None):
    with open(path_to_dataset) as f:
        data = pd.read_csv(f, skiprows=skip, header=None, nrows=96)
    print ("Data loaded from csv. Formatting...")
    result = np.array(data[2])
    print ("Data shape : ", result.shape)
    return result


# TODO corelations
def load_data_days_add():
    path_to_dataset = 'misc/91_trnava_suma_.csv'
    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=',')
        power = []
        nb_of_values = 0
        rng = pd.date_range(start='2013-07-01', end='2015-02-17', freq='15T')
        rng = rng[:-1]
        inc = 0
        for line in data:
            try:
                power.append(float(line[2]))
                if rng[inc].dayofweek is 5 or rng[inc].dayofweek is 6:
                    power.append(0)
                else:
                    power.append(1)
                nb_of_values += 1
                inc += 1
            except ValueError:
                pass
    print ("Data loaded from csv. Formatting...")

    result = []
    for index in np.arange(start=0, stop=114240, step=96 * 2):
        result.append(power[index: index + 96 * 2 * 2])

    result = np.array(result)
    result_mean = result[:, 0::2].mean()
    result_std = result[:, 0::2].std()
    result[:, 0::2] -= result_mean
    result[:, 0::2] /= result_std
    # result = zscore(result)

    print ("Data shape : ", result.shape)
    row = int(round(0.95 * result.shape[0]))

    train = result[:row, :]
    half = train.shape[1] / 2
    X_train = train[:, :half]
    y_train = train[:, half:]
    X_test = result[row:, :half]
    y_test = result[row:, half:]

    y_train = y_train[:, 0::2]
    y_test = y_test[:, 0::2]

    return [X_train, y_train, X_test, y_test]


def get_validation_data(x_tr, y_tr, ratio=0.1):
    validation_size = int(round(x_tr.shape[0] * ratio))
    ids = random.sample(range(x_tr.shape[0]), validation_size)
    X_val = x_tr[ids]
    y_val = y_tr[ids]
    x_tr = np.delete(x_tr, ids, axis=0)
    y_tr = np.delete(y_tr, ids, axis=0)
    return [x_tr, y_tr, X_val, y_val]
