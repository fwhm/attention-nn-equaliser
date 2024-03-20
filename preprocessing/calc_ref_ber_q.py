# Internal
from preprocessing.config.preprocessing_config import TEST_HDF5_DATASET_PATH
from utils.helper import get_ref_ber_q_from_hdf5
from global_config.global_config import LAUNCH_POWER_RANGE_LIST, MODULATION_ORDER, CROSS_VAL_DATA_PATH


# External
import numpy as np
import os.path

Plch_BER_Q = np.zeros([len(LAUNCH_POWER_RANGE_LIST), 3])
for idx, plch in enumerate(LAUNCH_POWER_RANGE_LIST):
    path_to_hdf5file = os.path.join(TEST_HDF5_DATASET_PATH, str(plch) + '_Pdbm.hdf5')
    Plch_BER_Q[idx] = np.append(plch, get_ref_ber_q_from_hdf5(path_to_hdf5file, MODULATION_ORDER))

np.save(os.path.join(CROSS_VAL_DATA_PATH, 'Plch_BER_Q.npy'), Plch_BER_Q)

