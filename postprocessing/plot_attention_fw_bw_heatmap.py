# https://stackoverflow.com/a/45842334/4724638

# Internal
from postprocessing.config.postprocessing_config import *
from global_config.global_config import N_TAPS, OUTPUT_COLLECTION_PATH
from postprocessing.util.utils import *
from postprocessing.util.plots import plot_weight_heatmap, plot_attention_score
# External
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# CONFIG [start]
plch_str = '6_Pdbm'
CASE = SIMPLE_ATTENTION_LAYER_CASE
ARGS_STR_UNI = ATTENTION_BIGRU_32
ARGS_STR_BI = ATTENTION_BIGRU_UNPACK_16
SAVE_FIG = True
# CONFIG [end]
path_to_output_uni = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR_UNI, plch_str)
path_to_output_bi = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR_BI, plch_str)
path_to_attention_weights_uni = os.path.join(path_to_output_uni, 'attention_weights')
path_to_attention_weights_bi = os.path.join(path_to_output_bi, 'attention_weights')

idx = [0, 180, 180]  # idx for unidir_ini, unidir, bidir
path_to_attention_weights = [path_to_attention_weights_uni, path_to_attention_weights_uni, path_to_attention_weights_bi]
att_weights_filenames = ["epoch_" + str(idx[i]) + '.npy' for i in np.arange(0, len(idx))]
att_weights_abs_files = [os.path.join(path_to_attention_weights[i], att_weights_filenames[i]) for i in np.arange(0, len(idx))]


for i in np.arange(0, len(idx)):
    # data to plot

    att_weights = np.load(att_weights_abs_files[i])
    if i < len(idx)-1:  # unidir
        if i == 0:
            title = "attention score initialization"
        else:
            title = "attention score after training"
        plot_weight_heatmap(att_weights, save=SAVE_FIG, n_taps=N_TAPS, title=title)

    else:  # last file is for bidir
        att_weights_fw = att_weights[:(2*N_TAPS+1)]
        att_weights_bw = att_weights[(2*N_TAPS+1):]
        # x = np.tile(np.arange(start=-N_TAPS, stop=N_TAPS + 1, step=1),2)
        x = np.arange(start=-N_TAPS, stop=N_TAPS + 1, step=1)

        # plot_weight_heatmap(att_weights_fw, n_taps=N_TAPS, title=["forward attention: epoch ", idx[i]])
        plot_weight_heatmap(att_weights_fw, save=SAVE_FIG, n_taps=N_TAPS, title="forward attention score")
        plot_weight_heatmap(att_weights_bw, save=SAVE_FIG, n_taps=N_TAPS, title="backward attention score")
        # stem plot for attention weights

        # fig, ax = plt.subplots()
        # markerline, stemlines, baseline = ax.stem(x, att_weights, markerfmt='C0.', use_line_collection=False)
        # # setting property of baseline with color red and linewidth 2
        # plt.setp(baseline, color='C0', linewidth=.2)  # color='r'

        plot_attention_score(att_weights, save=SAVE_FIG, n_taps=N_TAPS, axis_share='sharey')
        # Create two subplots and unpack the output array immediately

        plot_attention_score(att_weights, save=SAVE_FIG, n_taps=N_TAPS, axis_share='sharex')
        # ax2.set_title('backward attention score', fontsize="small")

        # # plot fw/bw scores in one axis
        # fig, ax = plt.subplots()
        # markerline, stemlines, baseline = ax.stem(att_weights, markerfmt='C0.', use_line_collection=False)
        # # setting property of baseline with color red and linewidth 2
        # plt.setp(baseline, color='C0', linewidth=.2)  # color='r'

        # def update_ticks(x, pos):
        #     if x == 0:
        #         return 'Mean'
        #     elif pos == 6:
        #         return 'pos is 6'
        #     else:
        #         return x
        #
        # ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

        # for i, label in enumerate(ax.get_xticklabels()):
        #     if i % 3 != 0:
        #         label.set_visible(False)
        # plt.xticks(np.arange(len(att_weights)), list(np.arange(len(att_weights))), fontsize='xx-small')  # Set text labels.
        # ax.set_xticks(np.arange(min(x),max(x),21))

        plt.show()

