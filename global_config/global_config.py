# Internal
from utils.utils import find_dir_path

# External
import os
import numpy as np

# Optical transmission system Configuration
SYSTEM_CONFIGURATION = "SSMF_DP_SC_resolution_1km_20_X_50_Km_30_GBd_64_QAM"  # also suggesting the project data path

# For pre-processing
LAUNCH_POWER = 2  # dBm for single launch power
LAUNCH_POWER_MIN = 0  # dBm for launch power range
LAUNCH_POWER_MAX = 0
LAUNCH_POWER_RANGE = str(LAUNCH_POWER_MIN) + "_to_" + str(LAUNCH_POWER_MAX) + "dBm"
LAUNCH_POWER_RANGE_ARRAY = np.arange(LAUNCH_POWER_MAX - LAUNCH_POWER_MIN + 1) + LAUNCH_POWER_MIN
LAUNCH_POWER_RANGE_LIST = list(LAUNCH_POWER_RANGE_ARRAY)

# Parameters to study
MODULATION_ORDER = int(SYSTEM_CONFIGURATION[SYSTEM_CONFIGURATION.find('QAM') - 3:SYSTEM_CONFIGURATION.find('QAM') - 1])
N_TAPS = 20
N_ADJACENT_SYMBOLS = 2 * N_TAPS + 1
GUARD_BAND = 10**3
N_FEATURES = 4  # x, y pols
# N_FEATURES = 2  # x-pol data for inputs only


# Dataset details
NUM_CROSS_VAL_FOLDS = 1

# Source data path
HOME_DIR = os.path.expanduser('~')
SOURCE_DATA_PATH = os.path.join(HOME_DIR, 'SOURCE_DATA')

# Preprocessed project data paths
PROJECT_PATH = find_dir_path('project')
PROJECT_DATA_PATH = find_dir_path('project_data')
DATASET_PATH = os.path.join(PROJECT_DATA_PATH, SYSTEM_CONFIGURATION)  # Path to current datasets to be used

# All data paths, containing pre-processed numpy data
ALL_DATA_PATH = os.path.join(DATASET_PATH, "all")
ALL_INPUTS_DATA_PATH = os.path.join(ALL_DATA_PATH, "inputs")
ALL_TARGETS_DATA_PATH = os.path.join(ALL_DATA_PATH, "targets")
ALL_GRP_TARGETS_DATA_PATH = os.path.join(ALL_DATA_PATH, "grp_targets")

# Cross-val data paths, containing pre-processed cross-val numpy data
CROSS_VAL_DATA_PATH = os.path.join(DATASET_PATH, "cross_validation")
CROSS_VAL_INPUTS_DATA_PATH = os.path.join(CROSS_VAL_DATA_PATH, "inputs")
CROSS_VAL_TARGETS_DATA_PATH = os.path.join(CROSS_VAL_DATA_PATH, "targets")
CROSS_VAL_GRP_TARGETS_DATA_PATH = os.path.join(CROSS_VAL_DATA_PATH, "grp_targets")

# Output path
OUTPUT_COLLECTION_PATH = os.path.join(HOME_DIR, "OUTPUT", SYSTEM_CONFIGURATION)
# OUTPUT_COLLECTION_PATH = os.path.join(PROJECT_PATH, "OUTPUT", SYSTEM_CONFIGURATION)

# Path to test data directory
TMP_DATA_PATH = os.path.join(PROJECT_PATH, 'tmp')


if __name__ == "__main__":
    pass
