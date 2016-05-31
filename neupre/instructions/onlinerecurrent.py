from .base import Base


class OnlineRecurrent(Base):
    def run(self):
        from neupre.backend.onlinelstm_backend import LstmBackend
        from .base import get_meta_values
        import matplotlib.pyplot as plt
        from os.path import basename, exists
        from os import makedirs
        import pandas as pd
        import numpy as np
        plt.style.use('ggplot')

        nettype = 'vanilla'
        # nettype = 'lstm'

        mean, std = get_meta_values(self.options['-f'][:-4])
        statspath = 'results/online/recurrent/%s/%s/' % (nettype, basename(self.options['-f'])[:-4])
        if not exists(statspath):
            makedirs(statspath)

        imp = LstmBackend(96 * 5, 200, 96, 0.01, 6, 3, 1, self.options['-f'], self.options['--buffsize'], mean, std, statspath)

        # TODO simtoend
        if self.options['--simsteps']:
            for dummy in xrange(int(self.options['--simsteps'])):
                imp.sim()

            first_prediction = pd.to_datetime('2013-07-01') + pd.DateOffset(days=int(self.options['--buffsize'])    )
            index = pd.date_range(start=first_prediction, periods=int(self.options['--simsteps']))
            plt.ioff()

            with open('%s/stats.txt' % statspath, 'a') as f:

                f.write("Working day mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_working_days_one96)))
                # f.write("Working day mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_working_days_multi)))

                f.write("Weekends mean one96 MAPE :%f\n" % np.mean(np.array(imp.mapes_holidays_one96)))
                # f.write("Weekends mean multi MAPE :%f\n" % np.mean(np.array(imp.mapes_holidays_multi)))

                f.write("Jan mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_jan_one96)))
                f.write("Feb mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_feb_one96)))
                f.write("Mar mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_mar_one96)))
                f.write("Apr mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_apr_one96)))
                f.write("May mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_may_one96)))
                f.write("Jun mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_jun_one96)))
                f.write("Jul mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_jul_one96)))
                f.write("Aug mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_aug_one96)))
                f.write("Sep mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_sep_one96)))
                f.write("Oct mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_oct_one96)))
                f.write("Nov mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_nov_one96)))
                f.write("Dec mean one96 MAPE: %f\n" % np.mean(np.array(imp.mapes_dec_one96)))

                # f.write("Jan mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_jan_multi)))
                # f.write("Feb mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_feb_multi)))
                # f.write("Mar mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_mar_multi)))
                # f.write("Apr mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_apr_multi)))
                # f.write("May mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_may_multi)))
                # f.write("Jun mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_jun_multi)))
                # f.write("Jul mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_jul_multi)))
                # f.write("Aug mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_aug_multi)))
                # f.write("Sep mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_sep_multi)))
                # f.write("Oct mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_oct_multi)))
                # f.write("Nov mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_nov_multi)))
                # f.write("Dec mean multi MAPE: %f\n" % np.mean(np.array(imp.mapes_dec_multi)))

            # maes
            plt.figure(figsize=(20, 4))
            # plt.plot(imp.maes_multi)
            # plt.plot(imp.maes_one96)
            series_maes_multi = pd.Series(data=imp.maes_multi, index=index)
            series_maes_one96 = pd.Series(data=imp.maes_one96, index=index)
            plt.plot(series_maes_multi, '--')
            plt.plot(series_maes_one96, '--')
            plt.ylabel('Mean Absolute Error')
            plt.savefig('%s/maes.png' % statspath)

            # mses
            plt.figure(figsize=(20, 4))
            # plt.plot(imp.mses_multi)
            # plt.plot(imp.mses_one96)
            series_mses_multi = pd.Series(data=imp.mses_multi, index=index)
            series_mses_one96 = pd.Series(data=imp.mses_one96, index=index)
            plt.plot(series_mses_multi, '--')
            plt.plot(series_mses_one96, '--')
            plt.ylabel('Mean Squared Error')
            plt.savefig('%s/mses.png' % statspath)

            # mapes
            plt.figure(figsize=(20, 4))
            # plt.plot(imp.mapes_multi)
            # plt.plot(imp.mapes_one96)
            series_mapes_multi = pd.Series(data=imp.mapes_multi, index=index)
            series_mapes_one96 = pd.Series(data=imp.mapes_one96, index=index)
            plt.plot(series_mapes_multi, '--')
            plt.plot(series_mapes_one96, '--')
            plt.ylabel('Mean Absolute Percentage Error')
            plt.savefig('%s/mapes.png' % statspath)
            plt.close('all')
        elif self.options['--simtoend']:
            pass
