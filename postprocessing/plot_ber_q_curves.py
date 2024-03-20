# internal
from postprocessing.config.postprocessing_config import *
from postprocessing.util.utils import *
from postprocessing.util.plots import plot_ber_q_from_np_dicts
from global_config.global_config import (
    PROJECT_PATH,
    SYSTEM_CONFIGURATION,
    LAUNCH_POWER_RANGE_LIST
)
# external
import os
import numpy as np


def get_legends(file_strs, params, **kwargs):
    # file_strs = kwargs.get('file_strs')
    # params = kwargs.get('params')
    legends = []
    for file_str in file_strs:
        legend_str = []
        for param in params:
            if param not in file_str:
                continue
            if param == 'att':
                legend_str += 'attention '
            if param == 'rnn_type':
                legend_str += get_rnn_type(file_str)
            if param == 'trim_out_tap':
                legend_str += get_trim_taps(file_str) + ' '
            if param == 'n_hiddens':
                legend_str += get_n_hiddens(file_str, 'str')
            if param == 'in_tap_ids':
                legend_str += get_tap_ids(file_str, type='in')
                # legend_str +=
            if param == 'out_tap_ids':
                legend_str += get_tap_ids(file_str, type='out')
                pass
            if param == 'bidirectional':
                if get_bidirectional(file_str) is True:
                    legend_str += 'Bi'
                else:
                    pass
            if param == 'trim':
                if get_trim(file_str) is True:
                    legend_str += ' trim'
                else:
                    pass
            if param == 'part':
                if get_part(file_str) is True:
                    legend_str += 'partial '
                else:
                    pass


        legend_str = ''.join(legend_str)
        legends.append(legend_str)
    return legends


path_to_np_BER_Q = os.path.join(PROJECT_PATH, 'postprocessing', SYSTEM_CONFIGURATION, 'Plch_BER_Q')
plot_reference = True
ATT = True
line_style = '-'
plot_type = 'q'
save_figure = True
poi = True
poi_str = ''
power_of_interest = np.arange(-2, 3)
plot_kw = {}
np_ber_q_dict = {}

file_strs = [
    # ATTENTION_BILSTM_8  # waiting for results
    RNN_BILSTM_32,
    # ATTENTION_BILSTM_32,
    # RNN_LSTM_64,

    # RNN_BILSTM_16,
    # RNN_LSTM_32,

    # RNN_BILSTM_8,
    # RNN_LSTM_16,


    # ATTENTION_BILSTM_16,



    # TRIM_RNN_TTAP1_BILSTM16,
    # TRIM_RNN_TTAP2_BILSTM16,


    # RNN_BILSTM_16,



    # RNN_BIGRU_16,
    RNN_BIGRU_32,
    # ATTENTION_BIGRU_16,
    # ATTENTION_BIGRU_32,
    # TRIM_RNN_TTAP1_BILSTM32,
    # TRIM_RNN_TTAP2_BILSTM32,
    # RNN_IN__1_2_OUT__1_2_BILSTM_32,
    # RNN_IN__3_3_OUT__1_2_BILSTM_32,
    # RNN_OUT__2_3_BILSTM_32,
    # RNN_OUT__5_5_BILSTM_32,
    # RNN_BIGRU_32,
    # TRIM_RNN_TTAP3_BIGRU32,
    # RNN_OUT__3_3_BIGRU_32,
    # PART_BILSTM_32_FW_3_BW_3,
    # PART_BIGRU_32_FW_3_BW_3,
    # PART_BIGRU_32_FW_3_BW_3_TRIM,
    # RNN_BIGRU_24,
    # RNN_BILSTM_21,
    PART_BIGRU_42_FW_3_BW_3_TRIM,
    PART_BIGRU_49_FW_3_BW_3_TRIM
    # RNN_IN__5_5_OUT__5_5_BILSTM_32,
    # RNN_OUT__2_2_BILSTM_32,


    # ATTENTION_LSTM_16,

    # ATTENTION_BILSTM_32,
    # RNN_BILSTM_32,
    # ATTENTION_LSTM_32,

    # ATTENTION_LSTM_64,

]
file_strs = list(map(lambda ls: ls+".npy", file_strs))
# # Option 1: legends For partial BRNN
# legends = get_legends(file_strs=file_strs, params=['part', 'bidirectional', 'rnn_type', 'n_hiddens', 'trim'])
# # Option 2: legends for attention comparison
# legends = get_legends(file_strs=file_strs, params=['att', 'bidirectional', 'rnn_type', 'n_hiddens'])
# # Option 3: legends for bidirectional and unidirectional
# legends = get_legends(file_strs=file_strs, params=['bidirectional', 'rnn_type', 'n_hiddens'])
# # Option 4: legends for trim out
# legends = get_legends(file_strs=file_strs, params=['part', 'bidirectional', 'rnn_type', 'n_hiddens', 'trim', 'out_tap_ids'])
# Option 5: legends for performance comparison under same complexity partial and full
legends = get_legends(file_strs=file_strs, params=['part', 'bidirectional', 'rnn_type', 'n_hiddens', 'trim'])
# params=['att', 'in_tap_ids', 'out_tap_ids', 'trim', 'trim_out_tap', 'bidirectional', 'rnn_type', 'n_hiddens']
# load training results
for idx, file_str in enumerate(file_strs):
    np_file = os.path.join(path_to_np_BER_Q, file_strs[idx])
    plch_ber_q_array = np.load(np_file)
    np_ber_q_dict[legends[idx]] = plch_ber_q_array

if plot_reference:
    # load reference BER_Q
    ref_np_file = os.path.join(path_to_np_BER_Q, 'Plch_BER_Q.npy')
    np_ber_q_dict['reference CDC'] = np.load(ref_np_file)


if poi is True:   # if only only plot values of power of interest
    for k, v in np_ber_q_dict.items():
        ind_start = int(np.min(power_of_interest) - np.min(v[:, 0]))
        n_value = len(power_of_interest)
        np_ber_q_dict[k] = np_ber_q_dict[k][ind_start:ind_start+n_value]

    # power_of_interest:
    poi_str = str(int(np.min(power_of_interest))) + "to" + str(int(np.max(power_of_interest))) + "pdbm"


# plot ber & q dicts
plot_ber_q_from_np_dicts(plot_type, **np_ber_q_dict, save=save_figure, line_style=line_style, attention=ATT, poi=poi_str)
