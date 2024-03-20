# https://stackoverflow.com/a/45842334/4724638

# Internal
from postprocessing.config.postprocessing_config import *
from global_config.global_config import N_TAPS, OUTPUT_COLLECTION_PATH
from postprocessing.util.utils import *
# External
import matplotlib.pyplot as plt
import numpy as np
import os

# CONFIG [start]
plch_str = '6_Pdbm'
CASE = SIMPLE_ATTENTION_LAYER_CASE
ARGS_STR = ATTENTION_BIGRU_32
# CONFIG [end]
path_to_output = os.path.join(OUTPUT_COLLECTION_PATH, CASE, ARGS_STR, plch_str)

path_to_attention_weights = os.path.join(path_to_output, 'attention_weights')
post_flat = True if 'post' in path_to_output else False

N_HIDDENS = get_n_hiddens(ARGS_STR)
# Find n_hiddens - for post flat attention [end]

# Find bidirectional - for post flat attention [start]
bidirectional = get_bidirectional(ARGS_STR)
# Find bidirectional - for post flat attention [end]
# num_files = len(os.listdir(path_to_attention_weights))  # all attention weights files
num_files = 3  # test the first num_files * 3 epochs
n_plots = 3  # number of att_weights plots to be created
div = num_files//n_plots  # interval of plotting attention weights, which is called here "training stages"

idx = [3 * (i * div) for i in np.arange(0, n_plots)]  # 3 is the interval of saving attention weights
idx.append(3*(num_files-1))
att_weights_filenames = ["epoch_" + str(idx[i]) + '.npy' for i in np.arange(0, len(idx))]

att_weights_abs_files = [os.path.join(path_to_attention_weights, att_weights_filenames[i]) for i in np.arange(0, len(idx))]

for i in np.arange(0, len(idx)):
    # data to plot
    att_weights = np.load(att_weights_abs_files[i])
    x = np.arange(start=-N_TAPS, stop=N_TAPS + 1, step=1)
    if post_flat:
        x = np.arange(start=-(2 * N_TAPS * N_HIDDENS + N_HIDDENS), stop=2 * N_TAPS * N_HIDDENS + N_HIDDENS, step=1) \
            if bidirectional else np.arange(start=-(N_TAPS + 1/2) * N_HIDDENS, stop=(N_TAPS + 1/2) * N_HIDDENS, step=1)

    # plotting
    plt.rcParams["figure.figsize"] = 5, 2
    plt.figure(dpi=1200)
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
    # plt.title('attention score at training stage #' + str(i))
    plt.title('attention score at epoch #' + str(idx[i]))
    plt.xlabel('symbol idx regarding to the centre')
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

    if i == 2:
        path = os.getcwd()
        filename = os.path.join(path, 'attention_at_epoch_6.svg')
        plt.savefig('attention at epoch#6', format='svg')
    plt.show()

'''
# For testing plotting single att_weights
att_weights = np.load(att_weights_abs_files[0])

plt.rcParams["figure.figsize"] = 5, 2
x = np.arange(start=-N_TAPS, stop=N_TAPS+1, step=1)

fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., 0, 1]
ax.imshow(att_weights[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
ax.set_yticks([])
ax.set_xlim(extent[0], extent[1])
ax2.plot(x, att_weights)
plt.tight_layout()
plt.show()
'''

