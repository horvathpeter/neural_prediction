import csv
import numpy as np
import random
import pandas as pd


def load_data(dataset=None,
              path='neupre/misc/data_/91_trnava_suma_.csv',
              ratio=1.0,
              sequence_length=96,
              shuffle=False,
              zscore=False,
              reshape=False):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        max_values = ratio * 57216
        power = power_from_file(path, max_values)
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

    # return [X_train, y_train, X_test, y_test, result_mean, result_std]
    return [X_train, y_train, X_test, y_test]


def load_data_corr(path='neupre/misc/data_/91_trnava_suma_.csv',
                   ratio=1.0,
                   sequence_length=6,
                   shuffle=False,
                   zscore=False,
                   reshape=False):
    max_values = ratio * 57216
    power = power_from_file(path, max_values)
    print ("Data loaded from csv. Formatting...")

    result = []
    for t in range(len(power) - 336 * 4):
        inp = []
        inp.append(power[t])
        inp.append(power[t + (336 * 4 - 24 * 4 * 8)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 7 - 1)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 7)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 6)])
        inp.append(power[t + (336 * 4 - 24 * 4)])
        inp.append(power[t + 336 * 4])
        result.append(inp)
    result = np.array(result)

    print ("Data shape : ", result.shape)

    if zscore:
        result_mean = result.mean()
        result_std = result.std()
        result -= result_mean
        result /= result_std
        print ("Shift : ", result_mean)

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

    # return [X_train, y_train, X_test, y_test]
    # return [X_train, y_train, X_test, y_test, result_mean, result_std]
    return [X_train, y_train, X_test, y_test]

def load_data_days(dataset=None,
                   path='neupre/misc/data_/91_trnava_suma_.csv',
                   ratio=1.0,
                   sequence_length=96,
                   shuffle=False,
                   zscore=False,
                   reshape=False):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        max_values = ratio * 57216
        power = power_from_file(path, max_values)
        print ("Data loaded from csv. Formatting...")

    result = []
    for index in np.arange(start=0, stop=len(power)-96, step=96):
        result.append(power[index: index + 96 * 2])

    # for index in range(len(power) - sequence_length*2):
    #     result.append(power[index: index + sequence_length*2])

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

    # return [X_train, y_train, X_test, y_test, result_mean, result_std]
    return [X_train, y_train, X_test, y_test]


def load_data_days_more(dataset=None,
                        path='neupre/misc/data_/91_trnava_suma_.csv',
                        ratio=1.0,
                        sequence_length=96,
                        shuffle=False,
                        zscore=False,
                        reshape=False):
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        max_values = ratio * 57216
        power = power_from_file(path, max_values)

    result = []
    for index in np.arange(start=0, stop=len(power)-96*6, step=96):
        result.append(power[index: index + 96 * 6])

    # for index in range(len(power) - sequence_length*2):
    #     result.append(power[index: index + sequence_length*2])

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

    three4 = 96*5
    X_train = train[:, :three4]
    y_train = train[:, three4:]
    X_test = result[row:, :three4]
    y_test = result[row:, three4:]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # return [X_train, y_train, X_test, y_test, result_mean, result_std]
    return [X_train, y_train, X_test, y_test]


def load_data_online(maxline,
                     dataset=None,
                     path='neupre/misc/data_zscore/8_ba_suma_zscore.csv',
                     reshape=False):
    power = None
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        power = power_from_file(path, maxline)
        print ("Data loaded from csv. Formatting...")

    result = []
    for index in range(len(power) - 96 + 1):
        result.append(power[index: index + 96])

    result = np.array(result)
    print ("Data shape: ", result.shape)

    X_train = result[:, :-1]
    y_train = result[:, -1]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return [X_train, y_train]


def load_data_corr_online(maxline,
                          dataset=None,
                          path='neupre/misc/data_zscore/91_trnava_suma_zscore.csv',
                          reshape=False):
    power = None
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        power = power_from_file(path, maxline)
        print ("Data loaded from csv. Formatting...")

    result = []
    for t in range(len(power) - 336 * 4):
        inp = []
        inp.append(power[t])
        inp.append(power[t + (336 * 4 - 24 * 4 * 8)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 7 - 1)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 7)])
        inp.append(power[t + (336 * 4 - 24 * 4 * 6)])
        inp.append(power[t + (336 * 4 - 24 * 4)])
        inp.append(power[t + 336 * 4])
        result.append(inp)
    result = np.array(result)
    print ("Data shape: ", result.shape)

    X_train = result[:, :-1]
    y_train = result[:, -1]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return [X_train, y_train]


def load_data_days_online(maxline,
                          dataset=None,
                          path='neupre/misc/data_zscore/8_ba_suma_zscore.csv',
                          reshape=False):
    power = None
    if isinstance(dataset, np.ndarray):
        power = dataset
        print ("Dataset by user. Formatting...")
    elif path:
        power = power_from_file(path, maxline)
        print ("Data loaded from csv. Formatting...")

    result = []
    # for index in np.arange(start=0, stop=maxline - 96, step=96):
    #     result.append(power[index: index + 96 * 2])

    for index in np.arange(start=0, stop=(maxline - 96 * 6)+96, step=96):
        result.append(power[index: index + 96 * 6])

    result = np.array(result)
    print ("Data shape : ", result.shape)

    three4 = 96 * 5
    X_train = result[:, :three4]
    y_train = result[:, three4:]

    if reshape:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return [X_train, y_train]


def load_data_point_online(maxline=96,
                           path='neupre/misc/data_zscore/8_ba_suma_zscore.csv',
                           skip=None):
    with open(path) as f:
        data = pd.read_csv(f, skiprows=skip, header=None, nrows=maxline)
    print ("Data loaded from csv. Formatting...")
    result = np.array(data[2])
    print ("Data shape : ", result.shape)
    return result


def power_from_file(path_to_dataset, maxline):
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
            if nb_of_values >= maxline:
                break
    return power


def get_validation_data(x_tr, y_tr, ratio=0.1):
    validation_size = int(round(x_tr.shape[0] * ratio))
    ids = random.sample(range(x_tr.shape[0]), validation_size)
    X_val = x_tr[ids]
    y_val = y_tr[ids]
    x_tr = np.delete(x_tr, ids, axis=0)
    y_tr = np.delete(y_tr, ids, axis=0)
    return [x_tr, y_tr, X_val, y_val]
