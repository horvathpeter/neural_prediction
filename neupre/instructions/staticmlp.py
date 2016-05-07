from .base import Base
from keras.callbacks import Callback


class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.loses_batch_end = []
        self.val = []

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.loses_batch_end.append(logs.get('loss'))

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass


class StaticMlp(Base):
    def run(self):
        from neupre.misc.dataops import load_data, load_data_corr
        from neupre.misc.dataops import load_data_days_more
        from neupre.misc.dataops import get_validation_data
        from neupre.misc.builders import build_model_mlp
        from .base import mape
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from .base import get_meta_values
        from os.path import basename, exists
        from os import makedirs

        plt.style.use('ggplot')
        mean, std = get_meta_values(self.options['-f'][:-4])

        if self.options['--onestep']:
            X_train, y_train, X_test, y_test = load_data(path=self.options['-f'], sequence_length=96)
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)

            model = build_model_mlp(inputs=95, hiddens=50, outputs=1)
            # train
            my_log = LossHistory()
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_data=(X_val, y_val), verbose=1,
                            callbacks=[my_log])



            statspath = 'misc/results/static/mlp/%s/' % basename(self.options['-f'])[:-4]
            if not exists(statspath):
                makedirs(statspath)

            # print log.history
            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
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
            # plt.show()
            plt.savefig('%s/onestep_prediction.png' % statspath)

        if self.options['--multistep']:
            X_train, y_train, X_test, y_test = load_data_days_more(path=self.options['-f'])
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)

            model_m = build_model_mlp(inputs=96 * 5, hiddens=200, outputs=96)
            # train
            my_log = LossHistory()
            log = model_m.fit(X_train, y_train, nb_epoch=100, batch_size=1, validation_data=(X_val, y_val), verbose=1,
                              callbacks=[my_log])

            statspath = 'misc/results/static/mlp/%s/' % basename(self.options['-f'])[:-4]
            if not exists(statspath):
                makedirs(statspath)

            # print log.history
            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            plt.savefig('%s/mses_multistep.png' % statspath)

            # predict
            y_pred = model_m.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1],))
            y_test = np.reshape(y_test, (y_test.shape[0] * y_test.shape[1],))

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
            # plt.show()
            plt.savefig('%s/multistep_prediction.png' % statspath)

        if self.options['--onestep96']:
            X_train, y_train, X_test, y_test = load_data_corr(path=self.options['-f'])
            X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train, 0.1)
            model = build_model_mlp(inputs=6, hiddens=5, outputs=1)
            # train
            # log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_split=0.1, verbose=2)
            log = model.fit(X_train, y_train, nb_epoch=10, batch_size=10, validation_data=(X_val, y_val), verbose=2)

            statspath = 'misc/results/static/mlp/%s/' % basename(self.options['-f'])[:-4]
            if not exists(statspath):
                makedirs(statspath)

            # print log.history
            plt.figure()
            plt.plot(log.history['loss'])
            plt.plot(log.history['val_loss'])
            # plt.show()
            # predict
            y_pred = model.predict(X_test)
            # reshape
            y_pred = np.reshape(y_pred, (y_pred.shape[0]))

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
                f.write("MAPE : %f\n\n" % mape(y_test, y_pred))

            print("MAPE ", mape(y_test, y_pred))

            plt.figure(figsize=(20, 4))
            plt.plot(y_test)
            plt.plot(y_pred)
            plt.savefig('%s/onestep96_prediction.png' % statspath)
