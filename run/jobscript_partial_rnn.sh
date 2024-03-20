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

# partial birnn specific
# '-fw_id', '--fw_id', default=0
# '-bw_id', '--bw_id', default=0



python3 partial_birnn.py -p=[0] -g=0 -rnn=lstm -n_h=16 
python3 partial_birnn.py -p=[0] -g=0 -rnn=lstm -n_h=32 
python3 partial_birnn.py -p=[0] -g=0 -rnn=lstm -n_h=64 

python3 partial_birnn.py -p=[0] -g=1 -rnn=lstm -n_h=32 
python3 partial_birnn.py -p=[0] -g=1 -rnn=lstm -n_h=16 
python3 partial_birnn.py -p=[0] -g=1 -rnn=lstm -n_h=08 

# GPU 0
python3 partial_birnn.py -p=[-3] -g=0 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[-2] -g=0 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[-1] -g=0 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[0] -g=0 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[1] -g=0 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3

# GPU 1
python3 partial_birnn.py -p=[2] -g=1 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[3] -g=1 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[4] -g=1 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[5] -g=1 -rnn=lstm -n_h=32 -fw_id=3 -bw_id=-3


python3 partial_birnn.py -p=[0] -g=0 -rnn=gru -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[2] -g=0 -rnn=gru -n_h=32 -fw_id=3 -bw_id=-3

python3 partial_birnn.py -p=[6] -g=0 -rnn=gru -n_h=32 -fw_id=3 -bw_id=-3 & python3 partial_birnn.py -p=[6] -g=0 -rnn=gru -n_h=32 -fw_id=2 -bw_id=-2
