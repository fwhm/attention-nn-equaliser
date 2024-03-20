"""overlapping attention weights for post flatten attention case"""
# internal
from global_config.global_config import OUTPUT_COLLECTION_PATH
from postprocessing.config.postprocessing_config import *
from postprocessing.util.utils import *

# external
import os
import numpy as np
import matplotlib.pyplot as plt

# CONFIG [start]
plch_str = '2_Pdbm'
CASE = POST_ATTENTION_LAYER_CASE
ARGS_STR = POST_ATTENTION_BILSTM_16
epoch = 300  # weights of epoch to be investigated
epoch = epoch//3*3
# CONFIG [end]
path_to_output = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR, plch_str)
path_to_attention_weights = os.path.join(path_to_output, 'attention_weights')
post_flat = True if 'post' in path_to_output else False
N_HIDDENS = get_n_hiddens(ARGS_STR)  # keep original n_hiddens, plots stacked for bidirectional case

n_tap = get_n_taps(ARGS_STR)
seq_len = 2 * n_tap + 1

att_weight_file = 'epoch_' + str(epoch) + '.npy'
att = np.load(os.path.join(path_to_attention_weights, att_weight_file))
att_reshape = []

for i in np.arange(0, seq_len):
    att_reshape.append(att[i * N_HIDDENS:(i+1)*N_HIDDENS])

# plt.figure(dpi=400)
hidden = np.arange(0, N_HIDDENS)
for att_weights in att_reshape:
    plt.stem(hidden, att_weights)
    # plt.scatter(hidden, att_weights, marker='.', linewidths=.5)
    # plt.plot(hidden, att_weights)

plt.xlabel('# hidden unit')
plt.ylabel('attention weights')
plt.title('Post-flatten attention weights on hidden units')
plt.show()
