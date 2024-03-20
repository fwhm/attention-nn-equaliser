"""
Plot Q factor, data stored in /home/OUTPUT/System_Config/NN_type/n_neuron/xPdBm/training_output
Read last three lines to get BER and Q values
Train loss: BER: Q:
Validation loss: BER: Q:
Test loss: BER: Q:
"""
# Internal
from global_config.global_config import (
    SYSTEM_CONFIGURATION,
    LAUNCH_POWER_RANGE_LIST
)
# External
import subprocess
import os
import matplotlib.pyplot as plt
import getpass
import socket
import re
import numpy as np


def tail1(f, n, offset=0):  # Not used now, TBD
    proc = subprocess.Popen(['tail', '-n', n + offset, f], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    return lines[:, -offset]


def tail(f, lines=20):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            f.seek(0,0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    return b'\n'.join(all_read_text.splitlines()[-total_lines_wanted:])


def get_BER_Q_from_last_n_lines(filename, n=5):
    """
    :param filename:
    :param n:
    :return: From a list containing strings of last n lines, get a list of Reference & Test BER and Q for plotting
    """
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    # last_n_lines = lines[-n:]
    ref_line, test_line = lines[-4], lines[-1]
    current_ber_index = [int(item) for item in [ref_line.rfind("BER"), test_line.rfind("BER")]]
    current_q_index = [int(item) for item in [ref_line.rfind("Q-Factor"), test_line.rfind("Q-Factor")]]
    # current_ber = [ref_line[current_ber_index[i]+4:current_q_index[i]] for i in np.arange(len(current_ber_index))]
    current_ber = [float(item) for item in [ref_line[current_ber_index[0]+len("BER:"):current_q_index[0]],
                                          test_line[current_ber_index[-1]+len("BER:"):current_q_index[-1]]]]
    current_q = [float(item) for item in [ref_line[current_q_index[0]+len("Q-Factor:"):],
                                        test_line[current_q_index[-1]+len("Q-Factor:"):]]]
    return current_ber, current_q


# Post-processing params
user_host = "liu2_y@WMGDS.WMG.WARWICK.AC.UK"
# output_path = "/home/" + getpass.getuser() + "@" + socket.gethostname().upper() + "/OUTPUT"
output_stored_path = "/home/" + user_host + "/OUTPUT"
system_config = SYSTEM_CONFIGURATION
n_neurons = 16

output_dir = os.path.join(output_stored_path, system_config)
plch_str = [str(plch) + "_Pdbm" for plch in LAUNCH_POWER_RANGE_LIST]
dict_ber, dict_q = {}, {}
ref_ber, train_ber, ref_q, train_q = [], [], [], []
for idx, nn_model in enumerate(os.listdir(output_dir)):
    results_path = os.path.join(output_dir, nn_model, "n_neurons_" + str(n_neurons))

    for plch in plch_str:
        training_output_file = os.path.join(results_path, plch, "training_output")
        # if idx == 0:
        current_ber, current_q = get_BER_Q_from_last_n_lines(training_output_file)
        if idx == 0:
            ref_ber.append(current_ber[0])
            ref_q.append(current_q[0])
        train_ber.append(current_ber[1])
        train_q.append(current_q[1])
    dict_ber[nn_model], dict_q[nn_model] = train_ber, train_q
    train_ber, train_q = [], []

# Dictionary containing data to be plotted
dict_ber["Reference"], dict_q["Reference"] = ref_ber, ref_q


def plot_ber(dict_ber, plch_list):
    for key, value in dict_ber.items():
        plt.plot(plch_list, value)
    plt.legend(list(dict_ber.keys()))
    plt.title('Power [dBm] VS BER')
    plt.xlabel('Power [dBm]')
    plt.ylabel('BER')
    plt.grid()
    plt.show()


def plot_q(dict_q, plch_list):
    for key, value in dict_q.items():
        plt.plot(plch_list, value)
    plt.legend(list(dict_ber.keys()))
    plt.title('Power [dBm] VS Q-Factor')
    plt.xlabel('Power [dBm]')
    plt.ylabel('Q-Factor [dB]')
    plt.grid()
    plt.show()





if __name__ == "__main__":
    # # ========== concatenate last few lines ========= #
    # path_to_file = "training_output_test"
    # file = open(path_to_file, 'rb')
    # a = tail(file, lines=2)
    # print(a)
    # # ========== end of last few lines ============ #

    # # ========== read lines into list ========= #
    # filename = "training_output_test"
    # lines = get_last_n_lines(filename, 3)
    # # ========== end of read lines into list ========= #
    a = 'oir 0.332 ii'
    # reout = re.findall(r'\d+', a)
    # idx = a.rfind('0.')
    # _ = get_BER_Q_from_last_n_lines("training_output_test")
    plot_ber(dict_ber, LAUNCH_POWER_RANGE_LIST)
    plot_q(dict_q, LAUNCH_POWER_RANGE_LIST)
    pass

