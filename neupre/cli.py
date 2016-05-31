"""
neupre

Usage:
  neupre staticmlp [--onestep --multistep --onestep96] [-f FILE]
  neupre staticrecurrent [--onestep --multistep --onestep96] [-f FILE]
  neupre onlinemlp --simsteps=<steps> [--buffsize=<days>] [-f FILE]
  neupre onlinerecurrent --simsteps=<steps> [--buffsize=<days>] [-f FILE]
  neupre -h | --help
  neupre --version

Options:
  -f FILE                           Time series file [default: neupre/misc/data_zscore/8_ba_suma_zscore.csv].
  --onestep                         One step ahead prediction (15 minutes ahead).
  --multistep                       Multi step ahead prediction (one day ahead).
  --buffsize=<days>                 Buffer size - size of the window with most recent values in days [default: 70].
  -h --help                         Show this screen.
  --version                         Show version.

Examples:
  neupre staticmlp --onestep --multistep --onestep96 -f neupre/misc/data_zscore/91_trnava_suma_zscore.csv
  neupre onlinemlp --simsteps=20 -f neupre/misc/data_zscore/91_trnava_suma_zscore.csv
  neupre onlinerecurrent --simsteps=100 --buffsize=50 -f neupre/misc/data_zscore/91_trnava_suma_zscore.csv

Help:
  For help using this tool, please open an issue on the Github repository:
  https://github.com/horvathpeter/neural_prediction
"""

from inspect import getmembers, isclass
from docopt import docopt

from . import __version__ as VERSION


def main():
    """Main CLI entrypoint."""
    import instructions
    options = docopt(__doc__, version=VERSION)

    # Here we'll try to dynamically match the command the user is trying to run
    # with a pre-defined command class we've already created.
    # print options
    for k, v in options.iteritems():
        if hasattr(instructions, k) and v:
            module = getattr(instructions, k)
            commands = getmembers(module, isclass)
            command = [command[1] for command in commands if command[0] != 'Base'][0]
            command = command(options)
            command.run()


# from neupre.instructions import StaticMlp, StaticRecurrent
# from neupre.instructions.onlinemlp import OnlineMlp
# from neupre.instructions.onlinerecurrent import OnlineRecurrent
#
# options = {'onlinerecurrent': True,
#            'onlinemlp': True,
#            'staticrecurrent': False,
#            'staticmlp': False,
#            '--multistep': True,
#            '--onestep': False,
#            '--onestep96': False,
#            '--simsteps': 100,
#            '--simtoend': False,
#            '--buffsize': 70,
#            '-f': 'misc/data_zscore/8_ba_suma_zscore.csv',
#            '--version': False,
#            '--help': False}
#

# from os import listdir
# from os.path import isfile, join

# data_path = "misc/data_zscore"
# files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[-3:] == 'csv']
# for datafile in files:
#     options['-f'] = join(data_path, datafile)
#     OnlineLstm(options).run()
#
# data_path = "misc/data_zscore"
# files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[-3:] == 'csv']
# for datafile in files:
#     options['-f'] = join(data_path, datafile)
# #     OnlineMlp(options).run()
#
# OnlineRecurrent(options).run()
# # OnlineMlp(options).run()

# maes = {}
# mses = {}
# mapes = {}
# data_path = "neupre/misc/data_zscore"
# files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[-3:] == 'csv']
# for datafile in files:
#     options['-f'] = join(data_path, datafile)
#     # mae, mse, mape = StaticMlp(options).run()
#     mae, mse, mape = StaticLSTM(options).run()
#     maes[datafile] = mae
#     mses[datafile] = mse
#     mapes[datafile] = mape
#
# StaticMlp(options).run()


#
#
from os import listdir
from os.path import isfile, join
# data_path = "misc/data_zscore"
# files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f[-3:] == 'csv']
# print files
# for datafile in files:
#     options['-f'] = join(data_path, datafile)
#     print datafile
#     sim_lstm = StaticLSTM(options)
#     sim_lstm.run()
#     # sim_mlp = StaticMlp(options)
#     # sim_mlp.run()
#     # sim_lstm = OnlineLstm(options)
#     # sim_lstm.run()
#     # sim_mlp = OnlineMlp(options)
#     # sim_mlp.run()


# from os.path import isfile, join
# from os.path import basename
# import pandas as pd
# import numpy as np
# #
# dataPath = 'misc/data_'
# files = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
#
#
# for csvfile in files:
#     fullfile = join(dataPath, csvfile)
#     ele = pd.read_csv(fullfile, header=None)
#     elenp = np.array(ele[2])
#     mean = elenp.mean()
#     std = elenp.std()
#     elenp -= mean
#     elenp /= std
#     ele[2] = elenp
#     metafile = 'misc/data_zscore/' + csvfile[:-4] + 'zscore_meta.txt'
#     datafile = 'misc/data_zscore/' + csvfile[:-4] + 'zscore.csv'
#
#     with open(metafile, mode='w') as f:
#         f.write('%f\n' % mean)
#         f.write('%f\n' % std)
#
#     ele.to_csv(datafile, header=False, index=False)
