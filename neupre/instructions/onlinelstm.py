from .base import Base


class OnlineLstm(Base):
    def run(self):
        from neupre.backend.onlinelstm_backend import Lstm
        imp = Lstm(96, 50, 50, 96, 0.01)

        # TODO simtoend
        if self.options['--simteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()
        elif self.options['--simtoend']:
            pass
