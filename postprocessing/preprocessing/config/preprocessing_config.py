# https://github.com/olly-styles/Trajectory-Tensors/blob/main/preprocessing/config/preprocessing_config.py
# System configuration
# path
from global_config.global_config import SOURCE_DATA_PATH
import os


SYSTEM_CONFIG = 'SSMF_DP_SC_resolution_1km_9_X_110_Km_34_GBd_64_QAM'


# Data paths
MERGE_HDF5_DATASET_PATH = os.path.join(SOURCE_DATA_PATH, SYSTEM_CONFIG, 'merge', 'hdf5files')

# Raw data path
TRAIN_MAT_DATASET_PATH = os.path.join(SOURCE_DATA_PATH, SYSTEM_CONFIG, 'matfiles', 'train')
TEST_MAT_DATASET_PATH = os.path.join(SOURCE_DATA_PATH, SYSTEM_CONFIG, 'matfiles', 'test')
TRAIN_HDF5_DATASET_PATH = os.path.join(SOURCE_DATA_PATH, SYSTEM_CONFIG, 'hdf5files', 'train')
TEST_HDF5_DATASET_PATH = os.path.join(SOURCE_DATA_PATH, SYSTEM_CONFIG, 'hdf5files', 'test')

