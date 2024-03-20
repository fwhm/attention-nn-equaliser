#!/bin/bash
# conda activate torch
# echo process ID:$$
# echo $$ > last_pid

# general params, config -p, -g
# '-p', '--plch', help="set of training powers in dB, e.g., [5] or [5,6,7]",default='[2]'
# '-t', '--n_taps', default='20'
# '-g', '--gpu_id', help='specify GPU', default='0'
# '-d', '--debug', help='t for debug and f for run', default='f'

# rnn specific
# '-rnn', '--rnn_type', help='gru or lstm', default='lstm'
# '-n_h', '--n_hiddens', help='n_hiddens for each layer', default='16'
# '-n_l', '--n_layers', help='number of rnn layers', default='1'
# '-bi', '--bidirectional', default='false'

# attention specific
# '-rs', '--return_sequence', default='true'


python3 rnn_model.py -p=[-3,-2,-1] -g=0 -rnn=lstm -n_h=16 -bi=false
python3 rnn_model.py -p=[-3,-2,-1] -g=0 -rnn=lstm -n_h=32 -bi=false
python3 rnn_model.py -p=[-3,-2,-1] -g=0 -rnn=lstm -n_h=64 -bi=false

python3 rnn_model.py -p=[-3,-2,-1] -g=1 -rnn=lstm -n_h=32 -bi=true
python3 rnn_model.py -p=[-3,-2,-1] -g=1 -rnn=lstm -n_h=16 -bi=true
python3 rnn_model.py -p=[-3,-2,-1] -g=1 -rnn=lstm -n_h=08 -bi=true

python3 rnn_model.py -p=[6] -g=0 -rnn=gru -n_h=32 -bi=true
