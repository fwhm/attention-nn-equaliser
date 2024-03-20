# Internal
from preprocessing.config.preprocessing_config import (
    TRAIN_HDF5_DATASET_PATH,
    TEST_HDF5_DATASET_PATH
)
from global_config.global_config import (
    LAUNCH_POWER_RANGE_LIST,

    N_ADJACENT_SYMBOLS,
    GUARD_BAND,

    # grp data save path
    ALL_DATA_PATH,
    ALL_INPUTS_DATA_PATH,
    ALL_TARGETS_DATA_PATH,
    ALL_GRP_TARGETS_DATA_PATH
)


from preprocessing.src.train_set_split import get_recv_symbols_data, get_sent_symbol_data, get_all_symbols_data
from utils.utils import check_makedir

# External
import h5py
import numpy as np
import os
import time
from tqdm import tqdm


# Define dataset constructor out of input raw data from .mat, return input data (x,y-pol)and desired output data (x-pol)
def create_dataset_from_hdf5(h5file, n_sym=N_ADJACENT_SYMBOLS, guard_band=GUARD_BAND):
    start = time.time()
    # read h5 dataset, shape=(none, 2), dtype=double
    with h5py.File(h5file, "r") as hf:

        recv_x = hf['recv']['xPol']
        recv_y = hf['recv']['yPol']
        sent_x = hf['sent']['xPol']
        sent_y = hf['sent']['yPol']

        # preprocess, discard symbols from both ends
        raw_size_discard = int(len(recv_x))
        range_discard = range(guard_band, int(raw_size_discard - guard_band))
        # Edit: following code only needed if using matlab to convert .mat to .h5
        # recv_x, recv_y, sent_x, sent_y = \
        #     np.hstack([recv_x[:, 0][range_discard].reshape(-1, 1), recv_x[:, 1][range_discard].reshape(-1, 1)]), \
        #     np.hstack([recv_y[:, 0][range_discard].reshape(-1, 1), recv_y[:, 1][range_discard].reshape(-1, 1)]), \
        #     np.hstack([sent_x[:, 0][range_discard].reshape(-1, 1), sent_x[:, 1][range_discard].reshape(-1, 1)]), \
        #     np.hstack([sent_y[:, 0][range_discard].reshape(-1, 1), sent_y[:, 1][range_discard].reshape(-1, 1)])

        # create dataset
        raw_size = recv_x.shape[0]
        dataset_size = raw_size - 2 * n_sym
        dataset_range = n_sym + np.arange(dataset_size)

        # x, y-pol input data
        data_recv = np.empty([dataset_size, n_sym, 4], dtype='float64')
        data_recv[:] = np.nan
        bnd_vec = int(np.floor(n_sym / 2))  # boundary for neighbouring symbols
        for vec_idx, center_vec in enumerate(dataset_range):
            # range of index for a set of neighbouring symbols centered at center_vec
            local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
            n_range = np.arange(0, n_sym)
            if np.any(local_range < 0) or np.any(local_range > raw_size):
                ValueError('Local range steps out of the data range during dataset creation!')
            else:
                data_recv[vec_idx, n_range, 0] = recv_x[local_range, 0]
                data_recv[vec_idx, n_range, 1] = recv_x[local_range, 1]
                data_recv[vec_idx, n_range, 2] = recv_y[local_range, 0]
                data_recv[vec_idx, n_range, 3] = recv_y[local_range, 1]

        if np.any(np.isnan(data_recv)):
            ValueError('Dataset matrix was not fully filled by data!')

        # x-pol desired output
        data_sent = np.empty([dataset_size, 2], dtype='float64')
        data_sent[:, 0] = sent_x[dataset_range, 0]
        data_sent[:, 1] = sent_x[dataset_range, 1]

        print('Loading h5 data used ' + "{:.2f}".format((time.time() - start)) + 's')
        return data_recv, data_sent


def create_dataset_from_inputs_targets(inputs, targets, n_sym=N_ADJACENT_SYMBOLS, guard_band=GUARD_BAND):
    start = time.time()
    recv_x, recv_y = inputs[0], inputs[1]
    sent_x, sent_y = targets[0], targets[1]
    # preprocess, discard symbols from both ends
    raw_size_discard = int(len(recv_x))
    range_discard = range(guard_band, int(raw_size_discard - guard_band))

    # create dataset
    raw_size = recv_x.shape[0]
    dataset_size = raw_size - 2 * n_sym
    dataset_range = n_sym + np.arange(dataset_size)

    # x, y-pol input data
    data_recv = np.empty([dataset_size, n_sym, 4], dtype='float64')
    data_recv[:] = np.nan
    bnd_vec = int(np.floor(n_sym / 2))  # boundary for neighbouring symbols
    for vec_idx, center_vec in enumerate(tqdm(dataset_range)):
        # range of index for a set of neighbouring symbols centered at center_vec
        local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
        n_range = np.arange(0, n_sym)
        if np.any(local_range < 0) or np.any(local_range > raw_size):
            ValueError('Local range steps out of the data range during dataset creation!')
        else:
            data_recv[vec_idx, n_range, 0] = recv_x[local_range, 0]
            data_recv[vec_idx, n_range, 1] = recv_x[local_range, 1]
            data_recv[vec_idx, n_range, 2] = recv_y[local_range, 0]
            data_recv[vec_idx, n_range, 3] = recv_y[local_range, 1]

    if np.any(np.isnan(data_recv)):
        ValueError('Dataset matrix was not fully filled by data!')

    # x-pol desired output
    data_sent = np.empty([dataset_size, 2], dtype='float64')
    data_sent[:, 0] = sent_x[dataset_range, 0]
    data_sent[:, 1] = sent_x[dataset_range, 1]

    print('Creating dataset used ' + "{:.2f}".format((time.time() - start)) + 's')
    return data_recv, data_sent


def create_grp_targets_dataset(targets, n_sym=N_ADJACENT_SYMBOLS, guard_band=GUARD_BAND):
    # Currently getting X-pol and Y-pol
    sent_x, sent_y = targets[0], targets[1]
    # preprocess, discard symbols from both ends
    raw_size_discard = int(len(sent_x))
    range_discard = range(guard_band, int(raw_size_discard - guard_band))

    # create dataset
    raw_size = sent_x.shape[0]
    dataset_size = raw_size - 2 * n_sym
    dataset_range = n_sym + np.arange(dataset_size)

    # x-pol target data
    data_sent = np.empty([dataset_size, n_sym, 4], dtype='float64')  # change 4 to 2 if want output size (N, L, 2)
    data_sent[:] = np.nan
    bnd_vec = int(np.floor(n_sym / 2))  # boundary for neighbouring symbols

    for vec_idx, center_vec in enumerate(tqdm(dataset_range)):
        # range of index for a set of neighbouring symbols centered at center_vec
        local_range = center_vec + np.arange(-bnd_vec, bnd_vec + 1)
        n_range = np.arange(0, n_sym)
        if np.any(local_range < 0) or np.any(local_range > raw_size):
            ValueError('Local range steps out of the data range during dataset creation!')
        else:
            data_sent[vec_idx, n_range, 0] = sent_x[local_range, 0]
            data_sent[vec_idx, n_range, 1] = sent_x[local_range, 1]
            data_sent[vec_idx, n_range, 2] = sent_y[local_range, 0]  # comment if want output size (N, L, 2)
            data_sent[vec_idx, n_range, 3] = sent_y[local_range, 1]  # comment if want output size (N, L, 2)

    if np.any(np.isnan(data_sent)):
        ValueError('Dataset matrix was not fully filled by data!')

    return data_sent


if __name__ == "__main__":
    # -- [start] Create merged grp_data before cross-val: generate train/test grp data separately then merge
    train_test_hdf5_data_paths = [TRAIN_HDF5_DATASET_PATH, TEST_HDF5_DATASET_PATH]
    all_np_save_path = [ALL_INPUTS_DATA_PATH, ALL_TARGETS_DATA_PATH, ALL_GRP_TARGETS_DATA_PATH]
    check_makedir(*all_np_save_path)
    for plch in LAUNCH_POWER_RANGE_LIST:
        plch_str = str(plch) + '_Pdbm'
        hdf5filename = plch_str + '.hdf5'
        inputs, targets, grp_targets = None, None, None
        for idx, path in enumerate(train_test_hdf5_data_paths):
            os.chdir(path)
            dict_recv = get_recv_symbols_data(hdf5filename)
            dict_sent = get_sent_symbol_data(hdf5filename)
            inputs_, targets_ = create_dataset_from_inputs_targets(list(dict_recv.values()), list(dict_sent.values()))
            grp_targets_ = create_grp_targets_dataset(list(dict_sent.values()))
            if inputs is None:
                inputs = inputs_
                targets = targets_
                grp_targets = grp_targets_
            else:
                inputs = np.concatenate([inputs, inputs_], axis=0)
                targets = np.concatenate([targets, targets_], axis=0)
                grp_targets = np.concatenate([grp_targets, grp_targets_], axis=0)
        np.save(os.path.join(ALL_DATA_PATH, "inputs", "inputs_" + plch_str + ".npy"), inputs)
        np.save(os.path.join(ALL_DATA_PATH, "targets", "targets_" + plch_str + ".npy"), targets)
        np.save(os.path.join(ALL_DATA_PATH, "grp_targets", "grp_targets_" + plch_str + ".npy"), grp_targets)
    # -- [end] Create merged grp_data -- #


    # # --------- [start] create train/ & test/ - grp_targets.npy datasets from hdf5 files --------- #
    # raw_hdf5_data_path = RAW_HDF5_DATASET_PATH
    # os.chdir(raw_hdf5_data_path)
    # check_makedir(TRAIN_GRP_TARGETS_DATA_PATH, TEST_GRP_TARGETS_DATA_PATH)
    # train_test_hdf5_data_paths = [RAW_HDF5_TRAIN_DATASET_PATH, RAW_HDF5_TEST_DATASET_PATH]
    # train_test_save_np_data_paths = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    # for idx, path in enumerate(train_test_hdf5_data_paths):
    #     for plch in LAUNCH_POWER_RANGE_LIST:
    #             plch_str = str(plch) + '_Pdbm'
    #             hdf5filename = plch_str + '.hdf5'
    #             os.chdir(path)
    #             dict_sent = get_sent_symbol_data(hdf5filename)
    #             grp_targets = create_grp_targets_dataset(list(dict_sent.values()))
    #
    #             np.save(os.path.join(train_test_save_np_data_paths[idx], "grp_targets", "grp_targets_" + plch_str + ".npy"), grp_targets)
    # # --------- [END] create train/ & test/ - grp_targets.npy datasets from hdf5 files --------- #
    # ##########################################################################################################
    # # --------- [start] create train/ & test/ - inputs.npy, targets.npy datasets from hdf5 files --------- #
    # raw_hdf5_data_path = RAW_HDF5_DATASET_PATH
    # os.chdir(raw_hdf5_data_path)
    # check_makedir(TRAIN_INPUTS_DATA_PATH, TRAIN_TARGETS_DATA_PATH, TEST_INPUTS_DATA_PATH, TEST_TARGETS_DATA_PATH)
    # train_test_hdf5_data_paths = [RAW_HDF5_TRAIN_DATASET_PATH, RAW_HDF5_TEST_DATASET_PATH]
    # train_test_save_np_data_paths = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    # for idx, path in enumerate(train_test_hdf5_data_paths):
    #     for plch in LAUNCH_POWER_RANGE_LIST:
    #         plch_str = str(plch) + '_Pdbm'
    #         hdf5filename = plch_str + '.hdf5'
    #         os.chdir(path)
    #         dict_recv = get_recv_symbols_data(hdf5filename)
    #         dict_sent = get_sent_symbol_data(hdf5filename)
    #         inputs, targets = create_dataset_from_inputs_targets(list(dict_recv.values()), list(dict_sent.values()))
    #         np.save(os.path.join(train_test_save_np_data_paths[idx], "inputs", "inputs_" + plch_str + ".npy"), inputs)
    #         np.save(os.path.join(train_test_save_np_data_paths[idx], "targets", "targets_" + plch_str + ".npy"), targets)
    #
    # # for plch in LAUNCH_POWER_RANGE_LIST:
    # #     plch_str = str(plch) + '_Pdbm'
    # #     hdf5filename = plch_str + '.hdf5'
    # #     dict_recv = get_recv_symbols_data(hdf5filename)
    # #     dict_sent = get_sent_symbol_data(hdf5filename)
    # #     dict_all = get_all_symbols_data(hdf5filename)
    # #
    # #     inputs, targets = create_dataset_from_inputs_targets(list(dict_recv.values()), list(dict_sent.values()))
    # #
    # #     np.save(os.path.join(MERGE_DATA_INPUTS_PATH, "inputs_" + plch_str + ".npy"), inputs)
    # #     np.save(os.path.join(MERGE_DATA_TARGETS_PATH, "targets_" + plch_str + ".npy"), targets)
    # #     # np.save(os.path.join(MERGE_DATA_ALL_PATH, "all" + ".npy"), list(dict_all))
    # # --------- [end] create train/ & test/ - inputs.npy, targets.npy datasets from hdf5 files --------- #
    ##########################################################################################################

    # raw_merge_data_path = RAW_MERGE_HDF5_DATASET_PATH
    # os.chdir(raw_merge_data_path)
    # # --------- [start] create targets_grp.npy datasets from hdf5 files --------- #
    # # targets_grp.npy contains grouped target data
    # check_makedir(MERGE_DATA_GRP_TARGETS_PATH)
    # for plch in LAUNCH_POWER_RANGE_LIST:
    #     plch_str = str(plch) + '_Pdbm'
    #     hdf5filename = plch_str + '.hdf5'
    #     dict_recv = get_recv_symbols_data(hdf5filename)
    #     grp_targets = create_grp_targets_dataset(list(dict_recv.values()))
    #     np.save(os.path.join(MERGE_DATA_GRP_TARGETS_PATH, "grp_targets_" + plch_str + ".npy"), grp_targets)
    # # --------- [end] create targets_grp.npy datasets from hdf5 files --------- #
    ##########################################################################################################
    # # --------- [start] create inputs.npy, targets.npy datasets from hdf5 files --------- #
    # # inputs.npy contains grouped input data, targets.npy contains only the center symbol
    #
    # if not os.path.exists(MERGE_DATA_INPUTS_PATH):
    #     os.makedirs(MERGE_DATA_INPUTS_PATH)
    # if not os.path.exists(MERGE_DATA_TARGETS_PATH):
    #     os.makedirs(MERGE_DATA_TARGETS_PATH)
    # if not os.path.exists(MERGE_DATA_ALL_PATH):
    #     os.makedirs(MERGE_DATA_ALL_PATH)
    # for plch in LAUNCH_POWER_RANGE_LIST:
    #     plch_str = str(plch) + '_Pdbm'
    #     hdf5filename = plch_str + '.hdf5'
    #     dict_recv = get_recv_symbols_data(hdf5filename)
    #     dict_sent = get_sent_symbol_data(hdf5filename)
    #     dict_all = get_all_symbols_data(hdf5filename)
    #
    #     inputs, targets = create_dataset_from_inputs_targets(list(dict_recv.values()), list(dict_sent.values()))
    #
    #     np.save(os.path.join(MERGE_DATA_INPUTS_PATH, "inputs_" + plch_str + ".npy"), inputs)
    #     np.save(os.path.join(MERGE_DATA_TARGETS_PATH, "targets_" + plch_str + ".npy"), targets)
    #     # np.save(os.path.join(MERGE_DATA_ALL_PATH, "all" + ".npy"), list(dict_all))
    # # --------- [end] create inputs.npy, targets.npy datasets from hdf5 files --------- #
    ##########################################################################################################
    pass
