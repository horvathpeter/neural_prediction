"""Base command"""
import numpy as np


class Base(object):
    def __init__(self, options, *args, **kwargs):
        self.options = options
        self.args = args
        self.kwargs = kwargs

    def run(self):
        raise NotImplementedError("The run() method is not implemented!")


def get_meta_values(zscorefile):
    """
    # get mean and std from meta file
    """
    meta_file = zscorefile + '_meta.txt'
    with open(meta_file) as f:
        lines = f.readlines()
    lines = [line[:-1] for line in lines]
    mean = float(lines[0])
    std = float(lines[1])
    return [mean, std]


def mape(real, pred):
    """Returns mean average percentage error"""
    return 100 * np.mean(np.abs((real - pred) / real))


def mape2(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return 100 * np.mean(np.abs(ypred[idx] - ytrue[idx]) / ytrue[idx])