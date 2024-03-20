# Internal
from global_config.global_config import (
    SYSTEM_CONFIGURATION,
    LAUNCH_POWER_RANGE_LIST,
)
# External
import os
from tqdm import tqdm
import numpy as np


def find(s, ch):  # find all the index of certain character 'ch' in a string 's'
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_all_training_loss(filename, n=5):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    training_loss = []
    for _, line in enumerate(tqdm(lines)):
        if 'Train loss' in line:
            idx = find(line, ':')
            training_loss.append(line[idx[0]+1:idx[1]-3])
    training_loss = np.asarray(training_loss)
    return training_loss


# path_to_output is to [plch + '_Pdbm']
path_to_output = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/torch/OUTPUT/' \
                 'SSMF_DP_SC_resolution_1km_20_X_50_Km_30_GBd_64_QAM/rnn/' \
                 'att_rnn_type_lstm_n_hiddens_32_n_layers_1_bidirectional_false_n_taps_20/5_Pdbm'

filename = 'training_output'

file = os.path.join(path_to_output, filename)

training_loss = get_all_training_loss(file)

np.save(os.path.join(path_to_output, "training_loss.npy"), training_loss)
