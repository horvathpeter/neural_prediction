from .base import Base


class OnlineMlp(Base):
    def run(self):
        from neupre.backend.onlinemlp_backend import MlpBackend
        from .base import get_meta_values
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        mean, std = get_meta_values(self.options['-f'][:-4])
        from os.path import basename, exists
        from os import makedirs
        statspath = 'misc/results/online/mlp/%s/' % basename(self.options['-f'])[:-4]
        if not exists(statspath):
            makedirs(statspath)

        imp = MlpBackend(96 * 5, 200, 96, 0.01, 6, 3, 1, self.options['-f'], self.options['--buffsize'], mean, std, statspath)

        # TODO simtoend
        if self.options['--simsteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()
            plt.ioff()
            plt.figure(figsize=(20, 4))
            plt.plot(imp.maes_multi)
            plt.plot(imp.maes_one96)
            plt.savefig('%s/maes.png' % statspath)
            plt.figure(figsize=(20, 4))
            plt.plot(imp.mses_multi)
            plt.plot(imp.mses_one96)
            plt.savefig('%s/mses.png' % statspath)
            plt.figure(figsize=(20, 4))
            plt.plot(imp.mapes_multi)
            plt.plot(imp.mapes_one96)
            plt.savefig('%s/mapes.png' % statspath)
            plt.close('all')
        elif self.options['--simtoend']:
            pass
