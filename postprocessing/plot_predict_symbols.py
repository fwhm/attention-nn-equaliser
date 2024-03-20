# Internal
from global_config.global_config import OUTPUT_COLLECTION_PATH
from postprocessing.config.postprocessing_config import *
from utils.utils import check_makedir
# External
import matplotlib.pyplot as plt
import numpy as np
import os

# CONFIG [start]
plch_str = '0_Pdbm'

save = True
# # opt 1 part BRNN
# CASE = PART_BRNN_CASE
# ARGS_STR = PART_BILSTM_32_FW_3_BW_3
# # opt 2 att
# CASE = SIMPLE_ATTENTION_LAYER_CASE
# ARGS_STR = ATTENTION_BILSTM_32
# # opt 3 rnn
CASE = RNN_CASE
ARGS_STR = RNN_BILSTM_32
# CONFIG [end]
path_to_output = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR, plch_str)
# NEXT LINE to be removed
path_to_output = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/OUTPUT/SSMF_DP_SC_resolution_1km_20_X_50_Km_30_GBd_64_QAM/att_mlp/mlp_depth_2_width_[100,100]/0_Pdbm'
# path_to_output = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/OUTPUT/SSMF_DP_SC_resolution_1km_9_X_110_Km_34_GBd_64_QAM/transformer_linear_encoder/mlp_depth_2_width_[16,16]/6_Pdbm'
pred_symbols_npy = os.path.join(path_to_output, 'pred_test_fold1.npy')
pred_symbols = np.load(pred_symbols_npy)
# plt.figure(dpi=600)
plt.scatter(pred_symbols[:, 0], pred_symbols[:, 1], marker='.', linewidths=.5)
plt.xlabel('I component')
plt.ylabel('Q component')
plt.title('Predicted symbols')

if save:
    savepath = os.path.join(os.getcwd(), 'plots', 'symbols')
    check_makedir(savepath)
    filename = os.path.join(savepath, 'predSymbols_rnn_bilstm32_0Pdbm.png')
    plt.savefig(filename, format='png')

plt.show()


def get_pred_symbols(nn_model):
    if nn_model == 'att':
        CASE = SIMPLE_ATTENTION_LAYER_CASE
        ARGS_STR = ATTENTION_BILSTM_32

    elif nn_model == 'rnn':
        CASE = RNN_CASE
        ARGS_STR = RNN_BILSTM_32

    path_to_output = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR, plch_str)
    pred_symbols_npy = os.path.join(path_to_output, 'pred_test_fold1.npy')
    pred_symbols = np.load(pred_symbols_npy)
    return pred_symbols
