# External
import os
import argparse
import time
import sys
import socket

import numpy as np
import random
import torch
from torchinfo import summary


# Ensure determinism in the results
def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_arg_str(args_dict, opt_list):
    arg_str = ""
    for i in range(len(opt_list)):
        arg_str += (opt_list[i] + '_')
        if i != len(opt_list) - 1:
            arg_str += (args_dict[opt_list[i]] + '_')
        else:
            arg_str += args_dict[opt_list[i]]

    return arg_str


def parsers():
    # for argparse
    parser = argparse.ArgumentParser(description='NN Equalisers')
    parser.add_argument('-p', '--plch', help="set of training powers in dB, e.g., [5] or [5,6,7]", default='[2]') # default='[0,1,2]'
    parser.add_argument('-n', '--n_neurons', help='same for encoder and decoder',
                        default='16')  # currently same number of neurons
    parser.add_argument('-g', '--gpu_id', help='specify GPU', default='0')
    parser.add_argument('-d', '--debug', help='t for True (debug) and f for False (run)', default='f')

    args = parser.parse_args()

    # for log_dir
    args_dict = vars(args)
    opt_list_plch = "plch,n_neurons".split(",")
    opt_list_wo_plch = "n_neurons".split(",")

    arg_str_plch = get_arg_str(args_dict, opt_list_plch)
    arg_str_wo_plch = get_arg_str(args_dict, opt_list_wo_plch)

    return args, arg_str_plch, arg_str_wo_plch


def get_plch_array(arg_str):
    p_start = arg_str.index('_[')
    p_end = arg_str.index(']_')
    p_str = arg_str[p_start + 1:p_end + 1]
    plch_array = np.asarray(eval(p_str))
    return plch_array


def check_makedir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


# Find the absolute path of
# 1. 'project', the current python project, assuming path is in the structure of /home/userx/PyCharmProjects/xxx
# #### 2. 'raw_data', assuming raw data is stored in desktop folder
# 3. 'project_data', data after preprocessing. Might be h5 data, stored under current project
# 0. get current server host name, to decide which folder to run project
def get_hostname():
    return socket.gethostname()


def find(s, ch):  # find all the index of certain character 'ch' in a string 's'
    return [i for i, ltr in enumerate(s) if ltr == ch]


def find_dir_path(dirname):  # choose from 'project', 'project_data', ['raw_data' removed],
    abspath_file = os.path.dirname(os.path.abspath(__file__))
    num_slash = abspath_file.count('/')  # get degree of current dir
    idx = find(abspath_file, '/')
    if dirname in ['project', 'project_data']:
        degree = 4
        if get_hostname() == 'vicrob' or get_hostname() == 'u1873231-System-Product-Name':
            degree -= 1
            if dirname == 'project':
                return abspath_file[:idx[degree]]

            elif get_hostname() == 'vicrob':
                return '/mnt/TB2/yifan/data'
            else:  # vicrob2
                return '/media2/yifan/data'

        if num_slash == degree:
            return abspath_file if dirname == 'project' else (os.path.expanduser('~') + '/data')
        else:
            return abspath_file[:idx[degree]] if dirname == 'project' else (os.path.expanduser('~') + '/data')

    else:
        sys.exit("Unrecognised directory name!")


# ======== [START] smart print model summary and training output, as well as write console print to file ======== #
def write_to_file():
    def wrapper(func):
        def deco(*args, **kwargs):
            func(*args, **kwargs)
            filename = os.path.join(kwargs['path'], func.__name__) \
                if 'path' in kwargs.keys() else sys.exit("Must pass a path to write to file")
            with open(filename, 'a') as f:
                sys.stdout = f
                func(*args, **kwargs)
                sys.stdout = sys.__stdout__
        return deco
    return wrapper


@write_to_file()
def model_summary(model, **kwargs):
    summary(model, input_size=kwargs['input_size'])


@write_to_file()
def training_output(**kwargs):
    '''

    :param kwargs: phase --> "train", "validation", "test", "reference"
                output = {'epoch_num':epoch} for printing epoch
                output = {'phase':phase, 'loss':loss, 'ber':ber, 'q':q} for printing output
                path = outputdir
    :return: None
    '''
    if len(kwargs['output']) == 1:  # 'epoch'
        print("----------- EPOCH " + str(list(kwargs['output'].values()))[1:-1] + " -----------")  # print epoch
        return

    if len(kwargs['output']) == 3:  # reference, no loss
        phase, ber, q = tuple(kwargs['output'].values())
        print(str.capitalize(phase) + " BER: {0:.6f} Q-Factor: {1:.2f}".format(ber, q))
        pass
    else:
        phase, loss, ber, q = tuple(kwargs['output'].values())
        print(str.capitalize(phase) + " loss: {0:.5f} BER: {1:.4f} Q-Factor: {2:.2f}".format(loss, ber, q))
# ======== [END] smart print model summary and training output, as well as write console print to file ======== #


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print("Running {} used {:.2f}s".format(func.__name__, time.time()-start))
    return wrapper


# ======== [START] TBD ======== #
def logger(func):
    def wrapper(*args, **kwargs):
        print('Start running {} function.'.format(func.__name__))
        func(*args, **kwargs)
        print('Finished.')
    return wrapper
# ======== [END] TBD ======== #


if __name__ == "__main__":
    print(find_dir_path('raw_data'))


# ======================= Deprecated functions ======================= #
'''
def parsers():
    # for argparse
    # parser = argparse.ArgumentParser("python3 nns_twc.py", description='NNs TWC')
    parser = argparse.ArgumentParser(description='NN Equalisers')
    parser.add_argument('-p', '--plch', help="set of training powers in dB, e.g., [5] or [5,6,7]", default='0')
    parser.add_argument('-att', '--attention_flag', help='if attention layer is added', default='y')
    parser.add_argument('-att_rtn', '--attention_return', help='if attention layer return_sequences=True', default='y')
    # if attention layers return sequences, if y, add bidirectional layers after attention layers
    parser.add_argument('-m', '--nn_model', help='nn model type, option: biLSTM, biGRU', default="biLSTM")
    parser.add_argument('-n', '--n_neurons', help='same for encoder and decoder',
                        default='16')  # currently same number of neurons
    parser.add_argument('-g', '--gpu_id', help='specify GPU', default='0')

    args = parser.parse_args()

    # for log_dir
    args_dict = vars(args)
    opt_list = "plch,n_neurons".split(",")
    arg_str = ""
    for i in range(len(opt_list)):
        arg_str += (opt_list[i] + '_')
        if i != len(opt_list) - 1:
            arg_str += (args_dict[opt_list[i]] + '_')
        else:
            arg_str += args_dict[opt_list[i]]

    return args, arg_str


def print_model_summary_to_file(model, input_size, output_save_path):
    summary(model, input_size=input_size)
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)

    file = output_save_path + '/' + "model_summary"
    original_stdout = sys.stdout
    with open(file, 'a') as f:
        sys.stdout = f
        summary(model, input_size=input_size)
        sys.stdout = original_stdout
    pass


def print_training_output_to_file(output_file_save_path, phase, ber, q):

    if not os.path.exists(output_file_save_path):
        os.makedirs(output_file_save_path)
    file = output_file_save_path + '/' + "training_output"
    original_stdout = sys.stdout
    with open(file, 'a') as f:
        sys.stdout = f
        if phase == "epoch":
            print('----epoch----')
        elif phase == "train":
            pass
        elif phase == "val":
            pass
        elif phase == "test":
            pass
        sys.stdout = original_stdout
        
'''

