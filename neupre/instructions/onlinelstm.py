from .base import Base


class OnlineLstm(Base):
    def run(self):
        from neupre.backend.onlinelstm_backend import LstmBackend
        import matplotlib.pyplot as plt


        meta_file = self.options['-f'][:-4] + '_meta.txt'
        with open(meta_file) as f:
            lines = f.readlines()
        lines = [line[:-1] for line in lines]
        mean = float(lines[0])
        std = float(lines[1])

        imp = LstmBackend(96 * 5, 200, 96, 0.01, 6, 3, 1, self.options['-f'], mean, std)

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
            plt.plot(imp.mapes_one96)
            plt.savefig('mapes.png')
        elif self.options['--simtoend']:
            pass

