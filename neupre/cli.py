"""
neupre

Usage:
  neupre staticmlp [--onestep | --multistep] [-f FILE]
  neupre staticlstm [--onestep | --multistep] [-f FILE]
  neupre onlinemlp (--simsteps=<steps> | --simtoend) [-f FILE]
  neupre onlinelstm
  neupre -h | --help
  neupre --version

Options:
  -f FILE                           Time series file [default: neupre/misc/data/91_trnava_suma_stand.csv].
  --onestep                         One step ahead prediction (15 minutes ahead)
  --multistep                       Multi step ahead prediction (one day ahead)
  -h --help                         Show this screen.
  --version                         Show version.

Examples:
  neupre staticmlp --onestep -f neupre/misc/data/91_trnava_suma.csv

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
    print options
    for k, v in options.iteritems():
        # print "k====> ", k
        # print "v====> ", v
        if hasattr(instructions, k) and v:
            module = getattr(instructions, k)
            commands = getmembers(module, isclass)
            command = [command[1] for command in commands if command[0] != 'Base'][0]
            command = command(options)
            command.run()

#
# optionse = {'--help': False,
#  '--version': False,
#  'hello': True,
#  'kokot': False,
#  'staticmlp':True}
#
# import neupre.instructions as instructions
# for k, v in optionse.iteritems():
#     # print "k====> ", k
#     # print "v====> ", v
#     print "hh"
#     if hasattr(instructions, k) and v:
#         print k
#         module = getattr(instructions, k)
#         print module
#         commands = getmembers(module, isclass)
#         print commands
#         print "prazdno"
#         command = [command[1] for command in commands if command[0] != 'Base'][0]
