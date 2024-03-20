import re


def find_substring_index(string, substring):
    return string.index(substring) + len(substring) - 1


def get_n_hiddens(args_str, type='int'):
    idx_hidden = find_substring_index(args_str, 'n_hiddens_')
    for idx_n_h, i in enumerate(args_str[idx_hidden + 1:]):
        if i == '_':
            break
    n_hiddens = args_str[idx_hidden + 1:idx_hidden + idx_n_h + 1]
    return n_hiddens if type == 'str' else int(n_hiddens)


def get_bidirectional(args_str):
    idx_bi = find_substring_index(args_str, 'bidirectional_')
    for idx_bi_end, i in enumerate(args_str[idx_bi + 1:]):
        if i == '_' or i == '.':
            break
    bidirectional = True if args_str[idx_bi + 1:idx_bi + idx_bi_end + 1] == 'true' else False
    return bidirectional


def get_rnn_type(args_str):
    idx_rnn = find_substring_index(args_str, 'rnn_type_')
    for idx_, i in enumerate(args_str[idx_rnn + 1:]):
        if i == '_':
            break
    rnn_type = args_str[idx_rnn + 1:idx_rnn + idx_ + 1].upper()
    return rnn_type


def get_n_taps(args_str):
    idx_n_taps = find_substring_index(args_str, 'n_taps_')
    for idx_, i in enumerate(args_str[idx_n_taps + 1:]):
        if i == '_':
            break
    n_taps = args_str[idx_n_taps + 1:idx_n_taps + idx_ + 2].upper()  # n_taps is the last arg, no '_' at the end
    return int(n_taps)


def get_trim_taps(args_str):
    idx_t_taps = find_substring_index(args_str, 'trim_out_tap_')
    for idx_, i in enumerate(args_str[idx_t_taps + 1:]):
        if i == '_':
            break
    trim_taps = args_str[idx_t_taps + 1:idx_t_taps + idx_ + 1]
    return 'trim_out_tap_' + trim_taps


def get_trim(args_str):
    idx_trim = find_substring_index(args_str, 'trim_')
    for idx_trim_end, i in enumerate(args_str[idx_trim + 1:]):
        if i == '_' or i == '.':
            break
    trim = True if args_str[idx_trim + 1:idx_trim + idx_trim_end + 1] == 'true' else False
    return trim


def get_part(args_str):
    return True if 'partial' in args_str else False


def get_tap_ids(args_str, type='out'):
    if type == 'in':
        idx_in_out_taps = find_substring_index(args_str, 'in_tap_ids_')
    else:
        idx_in_out_taps = find_substring_index(args_str, 'out_tap_ids_')
    str_start = args_str[idx_in_out_taps + 1:]
    # neg_count = 0
    for idx_, i in enumerate(str_start):
        if i == '_':
            # if str_start[idx_+1] != '_':
            #     break
            # else:
            break
    tap_ids = args_str[idx_in_out_taps + 1:idx_in_out_taps + idx_ + 1]
    # neg_count += 1

    # # tmp = [int(num) for num in re.findall(r'\d+', in_tap_ids)]
    # tmp = re.findall(r'\d+', tap_ids)
    # if neg_count == 0:
    #     tap_ids = tmp[0] + '_' + tmp[1]
    # elif neg_count == 1:
    #     tap_ids = '-' + tmp[0] + '_' + tmp[1]
    # elif neg_count == 2:
    #     tap_ids = '-' + tmp[0] + '_' + '-' + tmp[1]
    return ' in_' + tap_ids + ' ' if type == 'in' else ' out_' + tap_ids + ' '
    # return 'in_tap_ids_' + tap_ids + ' ' if type == 'in' else 'out_tap_ids_' + tap_ids + ' '
