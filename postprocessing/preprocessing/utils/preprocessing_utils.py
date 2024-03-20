# External
import numpy as np


def cplx2real(data_in):  # convert complex array to real array, input np array list, return tuple with multiple output
    data_out = []
    for _ in data_in:
        data_out.append(np.hstack([np.real(_), np.imag(_)]))
    return tuple(data_out)


def array2hdf5(array, hdf5filename):
    pass


def arrays2hdf5(arrays, hdf5filename):

    pass

