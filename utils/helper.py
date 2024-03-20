# External
import torch
import numpy as np
import h5py
from utils.ber_calc import qam_ber_gray, ber2q


def complex2real(x):  # convert complex array to real array, input np array list, return tuple with multiple output
    return np.hstack([np.real(x), np.imag(x)])


def real2complex(x):
    c = np.empty(len(x), dtype=np.complex64)
    c.real, c.imag = x[:, 0], x[:, 1]
    return c
    # return torch.complex(x) if working on tensors


def get_ref_ber_q_from_hdf5(path_to_h5py_file, mod_order=16):
    with h5py.File(path_to_h5py_file, 'r') as hf:
        data_symbols = real2complex(hf["recv"]["xPol"])
        ref_symbols = real2complex(hf["sent"]["xPol"])
        ref_ber = qam_ber_gray(data_symbols, ref_symbols, mod_order)
        ref_q = ber2q(ref_ber)
        return ref_ber, ref_q


def get_ref_ber_q_from_npy(path_to_npy_file, plch):
    plch_ber_q = np.load(path_to_npy_file)
    idx = np.argwhere(plch_ber_q[:, 0] == plch)
    return np.squeeze(plch_ber_q[idx, 1]), np.squeeze(plch_ber_q[idx, 2])
