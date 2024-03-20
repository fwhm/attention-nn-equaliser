# internal
# from postprocessing.plot_received_symbols import get_received_symbols
# from postprocessing.plot_predict_symbols import get_pred_symbols
# External
import matplotlib.pyplot as plt
import numpy as np
import inspect
import os
from pathlib import Path


def get_list(dict):
    return dict.keys


def check_makedir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def save_figure(absfilename, fig_format="svg"):
    plt.savefig(absfilename,format=fig_format)
    pass


def plot_ber_q_from_np_dicts(plot_type='q', save=False, line_style='', poi='', attention=False, **kwargs):
    # npy file shape [len(plch), 3], [plch, BER, Q]
    # dict: ['rnn_config_1': array_plch_ber_q_1, 'rnn_config_2': array_plch_ber_q_2, etc.]
    postprocessing_path = Path(os.getcwd())
    func_name = inspect.stack()[0][3]
    path_to_save = os.path.join(postprocessing_path, "plots", func_name)
    check_makedir(path_to_save)

    marker_style = ['s', 'D', 'v', '^', '>', 'p', 'o']
    plot_type = plot_type  # 'ber' or 'q'
    # metrics = kwargs.values()
    legends = kwargs.keys()
    # for _, value in kwargs.items():

    for i, (k, value) in enumerate(kwargs.items()):
        if plot_type == 'q':
            plt.plot(value[:, 0], value[:, 2], marker=marker_style[i], linestyle=line_style)
        elif plot_type == 'ber':
            plt.plot(value[:, 0], value[:, 1], marker=marker_style[i], linestyle=line_style)
            plt.yscale('log', base=10)

    plt.xlabel('Power [dBm]')
    if plot_type == 'q':
        plt.title('Power [dBm] VS Q-Factor')
        plt.ylabel('Q-Factor [dB]')
    elif plot_type == 'ber':
        plt.title('Power [dBm] VS BER')
        plt.ylabel('BER')
    plt.legend(legends)

    # opt1: major grid
    # plt.grid()
    # opt 2: minor grid
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='lightgray', linestyle='--')
    plt.tight_layout()
    # # inset axes for attention tilted plot
    # if attention is True:  # insert predicted symbols
    #     pred_symbols_rnn = get_pred_symbols('rnn')
    #     pred_symbols_att = get_pred_symbols('att')
    #     inset_ax1 = inset_axes(ax,
    #                     width="22%",  # width = 30% of parent_bbox
    #                     height=1.0,  # height : 1 inch
    #                     loc=1)
    #     plt.box(False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.scatter(pred_symbols_rnn[:, 0], pred_symbols_rnn[:, 1], marker='.', linewidths=.5)
    #     inset_ax2 = inset_axes(ax,
    #                        width="22%",  # width = 30% of parent_bbox
    #                        height=1.0,  # height : 1 inch
    #                        bbox_to_anchor=[0.4,0.6])
    #     plt.box(False)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.scatter(pred_symbols_att[:, 0], pred_symbols_att[:, 1], marker='.', linewidths=.5)
    if save is True:
        filename = plot_type.upper() + '_' + "_".join(legends)
        filename = filename.replace(' ', '_')
        if poi != '':
            filename = filename + "_" + poi
        absfilename = os.path.join(path_to_save, filename + ".svg")
        # absfilename = os.path.join(path_to_save, filename)
        save_figure(absfilename)

    plt.show()


def plot_weight_heatmap(att_weights, save=False, **kwargs):
    # For saving
    # postprocessing_path = Path(os.getcwd()).parent.absolute()
    postprocessing_path = Path(os.getcwd())
    func_name = inspect.stack()[0][3]
    path_to_save = os.path.join(postprocessing_path, "plots", func_name)
    check_makedir(path_to_save)

    plot_title = kwargs.get('title')
    N_TAPS = kwargs.get('n_taps')
    x = np.arange(start=-N_TAPS, stop=N_TAPS + 1, step=1)

    # plotting
    plt.rcParams["figure.figsize"] = 5, 2
    plt.figure(dpi=1200)
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    plt.title(plot_title)
    plt.xlabel('symbol idx regarding the centre')
    plt.ylabel('attention score')
    extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
    #  https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # ['twilight_shifted']
    ax.imshow(att_weights[np.newaxis, :], cmap='plasma', aspect="auto", extent=extent)
    ax.set_yticks([])
    ax.set_xlim(extent[0], extent[1])
    # ax2.plot(x, att_weights)
    # stem plot for attention weights
    markerline, stemlines, baseline = ax2.stem(x, att_weights, markerfmt='C0.', use_line_collection=False)
    # setting property of baseline with color red and linewidth 2
    plt.setp(baseline, color='C0', linewidth=.5)  # color='r'

    # markerfmt, first colour, style '.'
    plt.tight_layout()

    # save file
    if save is True:
        absfilename = os.path.join(path_to_save, plot_title+".svg")
        save_figure(absfilename)
    plt.show()


def plot_attention_score(att_weights, save=False, **kwargs):
    # For saving
    # postprocessing_path = Path(os.getcwd()).parent.absolute()
    postprocessing_path = Path(os.getcwd())
    func_name = inspect.stack()[0][3]
    path_to_save = os.path.join(postprocessing_path, "plots", func_name)
    check_makedir(path_to_save)

    axis_share = kwargs.get('axis_share')
    N_TAPS = kwargs.get('n_taps')
    x = np.arange(start=-N_TAPS, stop=N_TAPS + 1, step=1)
    att_weights_fw = att_weights[:len(att_weights)//2]
    att_weights_bw = att_weights[len(att_weights)//2:]

    if axis_share == 'sharex':  # vertical plot
        att_weights_bw = -att_weights_bw
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1_title = 'forward & backward attention score'
    elif axis_share == 'sharey':  # horizontal plot
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1_title = 'forward attention score'
        ax2_title = 'backward attention score'

    markerline, stemlines, baseline = ax1.stem(x, att_weights_fw)
    plt.setp(baseline, color='C0', markersize=.3, linewidth=.5)
    plt.setp(markerline, color='C0', markersize=2, linewidth=.5)
    ax1.set_title(ax1_title, fontsize="large")
    markerline, stemlines, baseline = ax2.stem(x, att_weights_bw)
    plt.setp(baseline, color='C0', markersize=1, linewidth=.5)
    plt.setp(markerline, color='C0', markersize=2, linewidth=.5)
    if axis_share == 'sharey':
        ax2.set_title(ax2_title, fontsize="large")

    if save is True:
        absfilename = os.path.join(path_to_save, axis_share+".svg")
        save_figure(absfilename)