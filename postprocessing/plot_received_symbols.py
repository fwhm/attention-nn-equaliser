# internal
from preprocessing.config.preprocessing_config import TRAIN_HDF5_DATASET_PATH
from preprocessing.src.train_set_split import get_recv_symbols_data
from utils.utils import check_makedir
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py


def get_received_symbols(pdbm):
    plch_file_str = str(pdbm)+'_Pdbm.hdf5'
    path_to_hdf5files = TRAIN_HDF5_DATASET_PATH
    hdf5file = os.path.join(path_to_hdf5files, plch_file_str)
    dict_recv = get_recv_symbols_data(hdf5file)
    recv_symbols = list(dict_recv.values())[0]  # For x-pol received data
    return recv_symbols


# CONFIG [start]
save = True
plch_file_str = '0_Pdbm.hdf5'
# CONFIG [end]
path_to_hdf5files = TRAIN_HDF5_DATASET_PATH
hdf5file = os.path.join(path_to_hdf5files, plch_file_str)

dict_recv = get_recv_symbols_data(hdf5file)
recv_symbols = list(dict_recv.values())[0]  # For x-pol received data
# plt.figure(dpi=600)
plt.scatter(recv_symbols[:, 0], recv_symbols[:, 1], marker='.', linewidths=.5)
plt.xlabel('I component')
plt.ylabel('Q component')
plt.title('Received symbols')


if save:
    savepath = os.path.join(os.getcwd(), 'plots', 'symbols')
    check_makedir(savepath)
    filename = os.path.join(savepath, 'rxSymbols_0Pdbm.png')
    plt.savefig(filename, format='png')

plt.show()

