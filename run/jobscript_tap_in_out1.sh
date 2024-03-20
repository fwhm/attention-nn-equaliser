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


python3 rnn_in_out_taps.py -p=[-3,-2] -g=1 -rnn=lstm -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-2:3] & python3 rnn_in_out_taps.py -p=[-1,0] -g=1 -rnn=lstm -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-2:3] & python3 rnn_in_out_taps.py -p=[1,2] -g=1 -rnn=lstm -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-2:3] & python3 rnn_in_out_taps.py -p=[3,4] -g=1 -rnn=lstm -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-2:3] & python3 rnn_in_out_taps.py -p=[5] -g=1 -rnn=lstm -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-2:3]

python3 rnn_in_out_taps.py -p=[-3,-2] -g=1 -rnn=gru -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-3:3] & python3 rnn_in_out_taps.py -p=[-1,0] -g=1 -rnn=gru -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-3:3] & python3 rnn_in_out_taps.py -p=[1,2] -g=1 -rnn=gru -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-3:3] & python3 rnn_in_out_taps.py -p=[3,4] -g=1 -rnn=gru -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-3:3] & python3 rnn_in_out_taps.py -p=[5] -g=1 -rnn=gru -n_h=32 -bi=true -i_ids=[-20:20] -o_ids=[-3:3]
