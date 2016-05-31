from .base import Base
# from keras.callbacks import Callback
#
#
# class LossHistory(Callback):
#     def __init__(self):
#         super(LossHistory, self).__init__()
#         self.loses_batch_end = []
#         self.val = []
#
#     def on_epoch_begin(self, epoch, logs={}):
#         pass
#
#     def on_epoch_end(self, epoch, logs={}):
#         pass
#
#     def on_batch_begin(self, batch, logs={}):
#         pass
#
#     def on_batch_end(self, batch, logs={}):
#         self.loses_batch_end.append(logs.get('loss'))
#
#     def on_train_begin(self, logs={}):
#         pass
#
#     def on_train_end(self, logs={}):
#         pass


class StaticRecurrent(Base):
    def run(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from neupre.misc.dataops import load_data
        from neupre.misc.dataops import load_data_days_more
        from neupre.misc.dataops import get_validation_data
        from neupre.misc.builders import build_model_lstm, build_model_lstm_simple, build_model_recurrent
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from .base import mape
        from .base import get_meta_values
        from os.path import basename, exists
        from os import makedirs
        plt.style.use('ggplot')

        nettype = 'vanilla'
        # nettype = 'lstm'
        mean, std = get_meta_values(self.options['-f'][:-4])

        if self.options['--onestep']:
            X_train, y_train, X_test, y_test = load_data(path=self.options['-f'], reshape=True)
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)

            # build
            # model = build_model_lstm(95, 50, 25, 1)
            model = build_model_recurrent(inputs=95, hiddens1=50, outputs=1)
            # model = build_model_lstm_simple(inputs=95, hiddens1=50, outputs=1)

            # train
            # log = model.fit(X_train, y_train, nb_epoch=10, batch_size=100, validation_split=0.1, verbose=2)
            # my_log = LossHistory()
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=100, validation_data=(X_val, y_val), verbose=1)

            statspath = 'results/static/recurrent/%s/%s/' % (nettype, basename(self.options['-f'])[:-4])
            if not exists(statspath):
                makedirs(statspath)

            # print log.history
            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.ylabel('Mean Squared Error')
            plt.savefig('%s/mses_onestep.png' % statspath)

            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            with open('%s/stats.txt' % statspath, 'w+'):
                pass
            with open('%s/stats.txt' % statspath, 'a') as f:
                f.write("Onestep Stats\n")
                f.write("MAE : %f\n" % mean_absolute_error(y_test, y_pred))
                f.write("MSE : %f\n" % mean_squared_error(y_test, y_pred))
                y_test *= std
                y_test += mean
                y_pred *= std
                y_pred += mean
                f.write("MAPE : %f\n\n" % mape(y_test, y_pred))

            print("MAPE ", mape(y_test, y_pred))

            plt.figure(figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.ylabel('Consumption (kW)')
            plt.savefig('%s/onestep_prediction.png' % statspath)

        if self.options['--multistep']:
            X_train, y_train, X_test, y_test = load_data_days_more(path=self.options['-f'], reshape=True)
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            # X_train, y_train, X_test, y_test, mean, std = load_data_days(path=self.options['-f'], reshape=True)

            # build
            # model = build_model_lstm_simple(96*5, 200, 96)
            model = build_model_recurrent(96*5, 200, 96)

            # train
            # my_log = LossHistory()
            # log = model.fit(X_train, y_train, nb_epoch=30, batch_size=20, validation_split=0.1, verbose=2)
            log = model.fit(X_train, y_train, nb_epoch=20, batch_size=10, validation_data=(X_val, y_val), verbose=1)

            statspath = 'results/static/recurrent/%s/%s/' % (nettype, basename(self.options['-f'])[:-4])
            if not exists(statspath):
                makedirs(statspath)

            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.ylabel('Mean Squared Error')
            plt.savefig('%s/mses_multistep.png' % statspath)

            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1]))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1]))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            with open('%s/stats.txt' % statspath, 'a') as f:
                f.write("Multistep Stats\n")
                f.write("MAE : %f\n" % mean_absolute_error(y_test, y_pred))
                f.write("MSE : %f\n" % mean_squared_error(y_test, y_pred))
                y_test *= std
                y_test += mean
                y_pred *= std
                y_pred += mean
                f.write("MAPE : %f\n\n" % mape(y_test, y_pred))

            print("MAPE ", mape(y_test, y_pred))

            plt.figure(figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.ylabel('Consumption (kW)')
            plt.savefig('%s/multistep_prediction.png' % statspath)

        if self.options['--onestep96']:
            from neupre.misc.dataops import load_data_corr
            X_train, y_train, X_test, y_test = load_data_corr(reshape=True,
                                                              path=self.options['-f'])

            # model = build_model_lstm_simple(6, 3, 1)
            model = build_model_recurrent(6, 3, 1)

            # train
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_split=0.1, verbose=1)

            statspath = 'results/static/recurrent/%s/%s/' % (nettype, basename(self.options['-f'])[:-4])
            if not exists(statspath):
                makedirs(statspath)

            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            plt.ylabel('Mean Squared Error')
            plt.savefig('%s/mses_onestep96.png' % statspath)

            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0],))

            print("MAE  ", mean_absolute_error(y_test, y_pred))
            print("MSE  ", mean_squared_error(y_test, y_pred))

            with open('%s/stats.txt' % statspath, 'a') as f:
                f.write("Onestep96 Stats\n")
                f.write("MAE : %f\n" % mean_absolute_error(y_test, y_pred))
                f.write("MSE : %f\n" % mean_squared_error(y_test, y_pred))
                y_test *= std
                y_test += mean
                y_pred *= std
                y_pred += mean
                f.write("MAPE : %f\n" % mape(y_test, y_pred))
            print("MAPE ", mape(y_test, y_pred))

            plt.figure(figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.ylabel('Consumption (kW)')
            plt.savefig('%s/onestep96_prediction.png' % statspath)
