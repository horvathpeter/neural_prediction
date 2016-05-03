from .base import Base
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class OnlineMlp(Base):
    def run(self):
        from neupre.backend.onlinemlp_backend import MlpBackend

        imp = MlpBackend(96 * 5, 200, 96, 0.01, self.options['-f'])

        # TODO simtoend
        if self.options['--simsteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()

            plt.figure(1, figsize=(20, 4))
            plt.plot(imp.maes_multi)
            plt.plot(imp.maes_one96)
            plt.savefig('maes.png')
            plt.figure(2, figsize=(20, 4))
            plt.plot(imp.mses_multi)
            plt.plot(imp.mses_one96)
            plt.savefig('mses.png')
            plt.figure(3, figsize=(20, 4))
            plt.plot(imp.mapes_multi)
            plt.savefig('mapes.png')
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
