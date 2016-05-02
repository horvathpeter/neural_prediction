from .base import Base


class OnlineLstm(Base):
    def run(self):
        from neupre.backend.onlinelstm_backend import LstmBackend
        import matplotlib.pyplot as plt
        # net_15min_ahead = build_model_lstm(95, 50, 50, 1)
        # net_day_ahead = build_model_lstm(96, 50, 50, 96)
        # net_day_ahead_96 = build_model_lstm(6, 3, 2, 1)
        # imp = Lstm(96, 50, 50, 96, 0.01)

        imp = LstmBackend(96 * 5, 200, 96, 0.01, self.options['-f'])

        # TODO simtoend
        if self.options['--simsteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()
            plt.figure(1, figsize=(20, 4))
            plt.plot(imp.maes)
            plt.savefig('maes.png')
            plt.figure(2, figsize=(20, 4))
            plt.plot(imp.mses)
            plt.savefig('mses.png')
            plt.figure(3, figsize=(20, 4))
            plt.plot(imp.mapes)
            plt.savefig('mapes.png')
        elif self.options['--simtoend']:
            pass

