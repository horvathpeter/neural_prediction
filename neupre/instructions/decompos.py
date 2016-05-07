import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from keras.layers.core import Dense
from keras.models import Sequential

from neupre.misc.dataops import load_data
from .base import Base

plt.style.use('ggplot')

decompfreq = 24 * 60 / 15 * 7


def build_model_ffnn():
    model = Sequential()
    model.add(Dense(
        output_dim=50,
        input_shape=(95,),
        activation='tanh'
    ))

    model.add(Dense(
        output_dim=1,
        activation='linear'
    ))
    start = time.time()
    print ("Compilin")
    model.compile(optimizer='SGD', loss='mse')
    print ("Compiled and took ", time.time() - start, "seconds")
    return model


def decompose():
    rng = pd.date_range(start='2013-07-01', end='2015-02-17', freq='15T')
    rng = rng[:-1]

    electric = pd.read_csv('data/91_trnava_suma_stand.csv', header=None)

    newd = pd.DataFrame(data=electric[2].tolist(), columns=['amount'])
    newd = newd.set_index(rng)

    newd.to_csv('data/91_trnava_suma_stand_df.csv')

    d = pd.read_csv('misc/data/91_trnava_suma_stand_df.csv',
                    names=['Datum', 'Amount'],
                    index_col=['Datum'],
                    parse_dates=True)

    dn = sm.tsa.seasonal_decompose(d.Amount.interpolate(), model='additive', freq=decompfreq)
    dn.plot()
    #
    # print (dn.observed)
    # print (dn.trend)
    # print (dn.seasonal)
    # print (dn.resid)

    # l = [dn.trend, dn.seasonal, dn.resid]
    # for index, comp in enumerate(l):
    #     l[index] = np.nan_to_num(comp)

    origin = d.Amount[decompfreq / 2:-decompfreq / 2]
    trend = (dn.trend[decompfreq / 2:-decompfreq / 2])
    # seasonal = np.nan_to_num(dn.seasonal)
    seasonal = (dn.seasonal[decompfreq / 2:-decompfreq / 2])
    residual = (dn.resid[decompfreq / 2:-decompfreq / 2])

    X_train_origin, y_train_origin, X_test_origin, y_test_origin = load_data(dataset=origin, shuffle=False,
                                                                             zscore=False)

    X_train_trend, y_train_trend, X_test_trend, y_test_trend = load_data(dataset=trend,
                                                                         shuffle=False,
                                                                         zscore=False)
    X_train_seasonal, y_train_seasonal, X_test_seasonal, y_test_seasonal = load_data(dataset=seasonal,
                                                                                     shuffle=False,
                                                                                     zscore=False)
    X_train_residual, y_train_residual, X_test_residual, y_test_residual = load_data(dataset=residual,
                                                                                     shuffle=False,
                                                                                     zscore=False)
    net_all = build_model_ffnn()
    net_trend = build_model_ffnn()
    net_seasonal = build_model_ffnn()
    net_residual = build_model_ffnn()

    net_all.fit(X_train_origin, y_train_origin, validation_split=0.1, nb_epoch=100, batch_size=100, verbose=2)
    net_trend.fit(X_train_trend, y_train_trend, validation_split=0.1, nb_epoch=100, batch_size=100, verbose=2)
    net_seasonal.fit(X_train_seasonal, y_train_seasonal, validation_split=0.1, nb_epoch=100, batch_size=100, verbose=2)
    net_residual.fit(X_train_residual, y_train_residual, validation_split=0.1, nb_epoch=100, batch_size=100, verbose=2)

    pred_origin = net_all.predict(X_test_origin)
    pred_trend = net_trend.predict(X_test_trend)
    pred_seasonal = net_seasonal.predict(X_test_seasonal)
    pred_residual = net_seasonal.predict(X_test_residual)

    pred_sum = pred_trend + pred_seasonal + pred_residual

    pred_trend = np.reshape(pred_trend, (pred_trend.shape[0]))

    plt.figure(figsize=(20, 3))
    plt.plot(y_test_trend)
    plt.plot(pred_trend)

    plt.figure(figsize=(20, 3))
    plt.plot(y_test_seasonal)
    plt.plot(pred_seasonal)

    plt.figure(figsize=(20, 3))
    plt.plot(y_test_residual)
    plt.plot(pred_residual)

    # result = []
    # for index in range(len(trend) - 96):
    #     result.append(trend[index: index + 96])
    # result = np.array(result)
    # result = zscore(result)
    #
    # row = int(round(0.90 * result.shape[0]))
    #
    # train = result[:row, :]
    # # np.random.shuffle(train)
    # X_trend = train[:, :-1]
    # y_t rend = train[:, -1]
    # X_trendt = result[row:, :-1]
    # y_trendt = result[row:, -1]
    # model.fit(X=X_trend, y=y_trend, nb_epoch=1, batch_size=100)
    #
    # p = model.predict(X_trendt)
    #
    # print ('%.2f' % dn.trend[0])

# decompose()



# dta = sm.datasets.co2.load_pandas().misc
# # deal with missing values. see issue
# dta.co2.interpolate(inplace=True)
#
# res = sm.tsa.seasonal_decompose(dta.co2, freq=decompfreq)
# resplot = res.plot()
#
# electric = pd.read_csv('data/91_trnava_suma_.csv', header=0, parse_dates=False)
# data = pd.read_csv('', skiprows=skip, header=None, nrows=96)
