# internal
from global_config.global_config import (
    OUTPUT_COLLECTION_PATH,
    PROJECT_PATH,
    SYSTEM_CONFIGURATION,
    LAUNCH_POWER_RANGE_LIST
)
from postprocessing.config.postprocessing_config import *
from utils.utils import find, check_makedir

# external
import os
import operator
import numpy as np

def get_reveresed_lines(filename):
    """
    :param filename:
    :param n:
    :return: From a list containing strings of last n lines, get a list of Reference & Test BER and Q for plotting
    """
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    lines = sorted(enumerate(lines), reverse=True)
    # https://stackoverflow.com/questions/12142133/how-to-get-first-element-in-a-list-of-tuples/12142903
    lines = [*map(operator.itemgetter(1), lines)]  # get certain element in a list of tuples

    return lines


def get_test_ber_q(lines):
    for line in lines:
        if 'Test loss' in line:
            idx_ber = line.find('BER:')
            idx_q = line.find('Q-Factor:')
            test_ber = float(line[idx_ber+4:idx_q])
            test_q = float(line[idx_q+9:])
            return test_ber, test_q


# CONFIG [start]
CASE = RNN_CASE
# SIMPLE_ATTENTION_LAYER_CASE  # TRIM_RNN_OUT_CASE  # RNN_IN_OUT_TAP_CASE  # RNN_CASE  # PART_BRNN_CASE
ARGS_STRS = [
    # RNN_BIGRU
    # RNN_BIGRU_16,
    # RNN_BIGRU_32,
    # PART_BILSTM_32_FW_3_BW_3,
    # PART_BIGRU_32_FW_3_BW_3_TRIM,
    # PART_BIGRU_32_FW_3_BW_3,
    # PART_BIGRU_42_FW_3_BW_3_TRIM,
    # PART_BIGRU_49_FW_3_BW_3_TRIM,
    RNN_BILSTM_21,
    RNN_BIGRU_24,

    # ATTENTION_BIGRU
    # ATTENTION_BIGRU_16,
    # ATTENTION_BIGRU_32
    # # RNN with taps
    # RNN_IN__1_2_OUT__1_2_BILSTM_32,
    # RNN_IN__3_3_OUT__1_2_BILSTM_32,
    # RNN_OUT__1_2_BILSTM_32,
    # RNN_OUT__2_3_BILSTM_32,
    # RNN_OUT__2_2_BILSTM_32,
    # RNN_OUT__5_5_BILSTM_32,
    # RNN_IN__5_5_OUT__5_5_BILSTM_32,
    # RNN_OUT__5_5_BIGRU_32,
    # RNN_OUT__3_3_BIGRU_32
    # # ATTENTION_BILSTM_32
    # # TRIM_RNN_TTAP1_BILSTM16,
    # # TRIM_RNN_TTAP2_BILSTM16,
    # # TRIM_RNN_TTAP1_BILSTM32,
    # # TRIM_RNN_TTAP2_BILSTM32,
]


# CONFIG [end]
output_paths = [os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR) for ARGS_STR in ARGS_STRS]

for output_path in output_paths:
    slash_idx = find(output_path, '/')
    args_str = output_path[slash_idx[-1]+1:]  # to be filled
    npy_save_path = os.path.join(PROJECT_PATH, 'postprocessing', SYSTEM_CONFIGURATION, 'Plch_BER_Q')
    check_makedir(npy_save_path)

    Plch_BER_Q = np.zeros([len(LAUNCH_POWER_RANGE_LIST), 3])
    Plch_BER_Q[:, 0] = LAUNCH_POWER_RANGE_LIST
    for idx, plch in enumerate(LAUNCH_POWER_RANGE_LIST):
        plch_str = str(plch)+'_Pdbm'
        training_output_txt = os.path.join(output_path, plch_str, 'training_output')
        if not os.path.exists(training_output_txt):
            continue  # initialise array as zeros, skip the results that are not present
        lines = get_reveresed_lines(training_output_txt)
        test_ber, test_q = get_test_ber_q(lines)
        Plch_BER_Q[idx] = np.asarray([plch, test_ber, test_q], dtype=float)

    npy_save_filename = os.path.join(npy_save_path, args_str)
    np.save(npy_save_filename, Plch_BER_Q)



