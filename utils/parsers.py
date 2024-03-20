import numpy as np
import argparse


class ArgsParser:
    """Base argsParser for NN equalisers"""
    def __init__(self, **kwargs):
        self.parser = argparse.ArgumentParser(description='NN Equalisers')
        self.add_general_args()
        self.add_case_args()
        self.args = self.parser.parse_args()
        # for log_dir
        self.args_dict = vars(self.args)
        self.opt_list = []
        self.init_opt_list()
        self.args_dir_str = self.get_args_dir_str()

    def add_general_args(self):
        self.parser.add_argument('-p', '--plch', help="set of training powers in dB, e.g., [5] or [5,6,7]",
                            default='[0]')  # default='[0,1,2]'
        self.parser.add_argument('-t', '--n_taps', default='20')
        self.parser.add_argument('-g', '--gpu_id', help='specify GPU', default='0')
        self.parser.add_argument('-d', '--debug', help='t for debug and f for run', default='f')

    def add_case_args(self):
        raise NotImplementedError

    def init_opt_list(self):
        raise NotImplementedError

    def get_args_dir_str(self):  # output folder structure according to case
        raise NotImplementedError

    def parsers(self):
        # parsers return args, plch_str and args_dir_str
        return self.args, self.args_dir_str


class RnnParser(ArgsParser):
    def __init__(self, **kwargs):
        super(RnnParser, self).__init__(**kwargs)

    def add_case_args(self):
        self.parser.add_argument('-rnn', '--rnn_type', help='gru or lstm', default='gru')
        self.parser.add_argument('-n_h', '--n_hiddens', help='n_hiddens for each layer', default='16')
        self.parser.add_argument('-n_l', '--n_layers', help='number of rnn layers', default='1')
        self.parser.add_argument('-bi', '--bidirectional', default='true')
        self.parser.add_argument('-up_bi', '--unpack_bidir', help='unpack fw/bw birnn output', default='false') # change default to false

    def init_opt_list(self):
        self.opt_list = "rnn_type,n_hiddens,n_layers,bidirectional,n_taps".split(",")
        if self.args.bidirectional == 'true':
            if self.args.unpack_bidir == 'true':
                self.opt_list.remove('bidirectional')
                self.opt_list.append('unpack_bidir')
        if self.args.n_layers == '1':
            self.opt_list.remove("n_layers")
        if self.args.n_taps == '20':
            self.opt_list.remove("n_taps")

    def get_args_dir_str(self):
        return get_arg_str(self.args_dict, self.opt_list)


class PartialBiRnnParser(RnnParser):
    def __init__(self, **kwargs):
        super(PartialBiRnnParser, self).__init__(**kwargs)

    def add_case_args(self, **kwargs):
        super(PartialBiRnnParser, self).add_case_args(**kwargs)
        self.parser.add_argument('-part', '--partial', default='')
        self.parser.add_argument('-fw_id', '--fw_id', help='forward rnn seq len, id regarding to tap0, including itself',
                                 default='3')
        self.parser.add_argument('-bw_id', '--bw_id',
                                 help='backward rnn seq len, id regarding to tap0, including itself',
                                 default='-3')
        self.parser.add_argument('-trim', '--trim',
                                 help='Extra trim for brnn output, false for no extra trim', default='false')


    def init_opt_list(self):
        # bidirectional but partially
        super(PartialBiRnnParser, self).init_opt_list()
        self.opt_list.insert(0, 'partial')
        self.opt_list.append('fw_id')
        self.opt_list.append('bw_id')
        self.opt_list.append('trim')


class InOutTapRnnParser(RnnParser):
    def __init__(self, **kwargs):
        super(RnnParser, self).__init__(**kwargs)

    def add_case_args(self, **kwargs):
        super(InOutTapRnnParser, self).add_case_args(**kwargs)
        self.parser.add_argument('-i_ids', '--in_tap_ids', default='[-20:20]')  # default -1,0,1,2
        self.parser.add_argument('-o_ids', '--out_tap_ids', default='[-1:2]')  # default -1,0,1,2

    def init_opt_list(self):
        super(InOutTapRnnParser, self).init_opt_list()
        self.opt_list.append('in_tap_ids')
        self.opt_list.append('out_tap_ids')


class ConvRnnParser(InOutTapRnnParser):
    def __init__(self, **kwargs):
        super(ConvRnnParser, self).__init__(**kwargs)

    def add_case_args(self):
        super(ConvRnnParser, self).add_case_args()
        self.parser.add_argument('-cnn', '--cnn', default='')
        self.parser.add_argument('-k', '--kernel_size', default='10')
        self.parser.add_argument('-n_f', '--n_filters', default='128')

    def init_opt_list(self):
        super(ConvRnnParser, self).init_opt_list()
        self.opt_list.insert(0, 'cnn')
        self.opt_list.insert(1, 'kernel_size')
        self.opt_list.insert(2, 'n_filters')


class TrimRnnOutParser(RnnParser):
    def __init__(self, **kwargs):
        super(TrimRnnOutParser, self).__init__(**kwargs)

    def add_case_args(self):
        super(TrimRnnOutParser, self).add_case_args()
        self.parser.add_argument('-t_tap', '--trim_out_tap', default='1')

    def init_opt_list(self):
        super(TrimRnnOutParser, self).init_opt_list()
        self.opt_list.insert(0, 'trim_out_tap')


class AttRnnParser(RnnParser):
    """
    Base template parsers for attention based on RNN,
    also suitable for single simplified attention layer case
    """
    def __init__(self, **kwargs):
        super(AttRnnParser, self).__init__(**kwargs)

    def add_case_args(self):
        super(AttRnnParser, self).add_case_args()
        # self.parser.add_argument('-head', '--num_heads', default='1')
        self.parser.add_argument('-att', '--att', default='')
        self.parser.add_argument('-rs', '--return_sequence', default='true')

    def init_opt_list(self):
        super(AttRnnParser, self).init_opt_list()
        self.opt_list.insert(0, 'att')


class SimpAttLayerRnnParser(AttRnnParser):
    """Parsers for single simplified attention layer based on RNN"""
    def __init__(self, **kwargs):
        super(SimpAttLayerRnnParser, self).__init__(**kwargs)

    def add_case_args(self):
        super(SimpAttLayerRnnParser, self).add_case_args()


class SimpSDCAttRnnParser(AttRnnParser):
    """Parsers for simplified attention based on RNN"""
    def __init__(self, **kwargs):
        super(SimpSDCAttRnnParser, self).__init__(**kwargs)

    def add_case_args(self):
        super(SimpSDCAttRnnParser, self).add_case_args()
        self.parser.add_argument('-head', '--n_heads', default='1')

    def init_opt_list(self):
        super(SimpSDCAttRnnParser, self).init_opt_list()
        self.opt_list.insert(1, 'n_heads')  # index 1, after att


class TransformerParser(ArgsParser):
    def __init__(self):
        super(TransformerParser, self).__init__()

    def add_case_args(self):
        self.parser.add_argument('-tf', '--transformer', default='')
        self.parser.add_argument('-emb', '--embed', default='true')
        self.parser.add_argument('-embsz', '--embed_size', default='512')
        self.parser.add_argument('-nl', '--n_layers', default='6')
        self.parser.add_argument('-fexp', '--forward_exp', default='4')
        self.parser.add_argument('-heads', '--heads', default='8')
        self.parser.add_argument('-dp', '--dropout', default='0')

    def init_opt_list(self):
        self.opt_list = "transformer,embed,n_layers,forward_exp,heads,dropout".split(",")
        if self.args.embed == 'true':  # if embedding, add only embed_size
            self.opt_list.insert(1, 'embed_size')
            self.opt_list.remove('embed')

    def get_args_dir_str(self):  # output folder structure according to case
        return get_arg_str(self.args_dict, self.opt_list)


class TransformerEncoderParser(TransformerParser):
    def __init__(self):
        super(TransformerEncoderParser, self).__init__()

    def add_case_args(self):
        super(TransformerEncoderParser, self).add_case_args()
        self.parser.add_argument('-enc', '--encoder', default='')
        

    def init_opt_list(self):
        super(TransformerEncoderParser, self).init_opt_list()
        self.opt_list.insert(1, 'encoder')


class MlpParser(ArgsParser):
    def __init__(self):
        super(MlpParser, self).__init__()

    def add_case_args(self):
        self.parser.add_argument("-mlp", "--mlp", default='')
        self.parser.add_argument("-dep","--depth", default='2', help="Does not count in the input layer and output layer "
                                                                     "as they depend on the input and output size")
        self.parser.add_argument("-w", "--width", default='[100,100]')

    def init_opt_list(self):
        self.opt_list = "mlp,depth,width".split(",")

    def get_args_dir_str(self):
        return get_arg_str(self.args_dict, self.opt_list)


def get_arg_str(args_dict, opt_list):
    arg_str = ""
    for i in range(len(opt_list)):
        arg_str += (opt_list[i] + '_')
        if i != len(opt_list) - 1:
            if args_dict[opt_list[i]] == '':
                arg_str += args_dict[opt_list[i]]
                continue
            arg_str += (args_dict[opt_list[i]] + '_')
        else:
            arg_str += args_dict[opt_list[i]]

    return arg_str


def get_arg_array(arg_str):
    arg_start = arg_str.index('[')
    arg_end = arg_str.index(']')
    if ':' in arg_str:
        idx_split = arg_str.index(':')
        arg_array = np.arange(int(arg_str[arg_start+1:idx_split]), int(arg_str[idx_split+1:arg_end])+1)
        return arg_array
    arg_str = arg_str[arg_start:arg_end + 1]
    arg_array = np.asarray(eval(arg_str))
    return arg_array


def get_args_dir_str(args_dict, opt_list):  # organise folder structure for outputs
    # replace '_' with '/'
    arg_str = ""
    for i in range(len(opt_list)):
        arg_str += (opt_list[i] + '_')
        if i != len(opt_list) - 1:
            arg_str += (args_dict[opt_list[i]] + '/')
        else:
            arg_str += args_dict[opt_list[i]]

    return arg_str



