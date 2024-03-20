# For hdf5 file, decided to work on numpy data, TBD, will come back later
# Internal
import h5py

from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    LAUNCH_POWER_RANGE_LIST,
    CROSS_VAL_INPUTS_DATA_PATH,
    CROSS_VAL_TARGETS_DATA_PATH,
    CROSS_VAL_GRP_TARGETS_DATA_PATH,
    TMP_DATA_PATH
)

# External
import numpy as np
import os
import pickle


def split_numpy_array(array, train_indexs, val_indexs, test_indexs):
    """
    Splits a numpy array into train, val, and test
    """
    train_data = array[train_indexs]
    val_data = array[val_indexs]
    test_data = array[test_indexs]
    return train_data, val_data, test_data


def split_numpy_arrays(array_list, train_index, val_indexs, test_indexs):
    """
        Splits a numpy array list into train, val, and test
    """
    train_data, val_data, test_data = ([], [], [])
    for array in array_list:
        train, val, test = split_numpy_array(array, train_index, val_indexs, test_indexs)
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)
    return train_data, val_data, test_data


def save_numpy_data(save_path, train_data, val_data, test_data, fold=None,):
    """
    Saves train val test numpy data
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if fold is None:
        np.save(os.path.join(save_path, "train" + ".npy"), train_data)
        np.save(os.path.join(save_path, "val" + ".npy"), val_data)
        np.save(os.path.join(save_path, "test" + ".npy"), test_data)
    else:
        np.save(os.path.join(save_path, "train_fold" + str(fold) + ".npy"), train_data)
        np.save(os.path.join(save_path, "val_fold" + str(fold) + ".npy"), val_data)
        np.save(os.path.join(save_path, "test_fold" + str(fold) + ".npy"), test_data)

def save_pickle_data(save_path, fold, train_data, val_data, test_data):
    """
    Saves train val test data as pickles
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "train_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(train_data, fp)
    with open(os.path.join(save_path, "val_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(val_data, fp)
    with open(os.path.join(save_path, "test_fold" + str(fold) + ".pickle"), "wb") as fp:
        pickle.dump(test_data, fp)


def save_hdf5_data(save_path, fold, train_data, val_data, test_data):
    """
    Saves train val test data as hdf5, data in the form of np array list for xPol and yPol
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "train_fold" + str(fold) + ".hdf5"), "w") as fh5:
        # grp_recv = f.create_group('recv')
        # recv_x_pol = grp_recv.create_dataset("xPol", data=train_data[0])
        # recv_y_pol = grp_recv.create_dataset("yPol", data=train_data[1])
        #
        # grp_sent = f.create_group('sent')
        # sent_x_pol = grp_sent.create_dataset("xPol", data=sent_x)
        # sent_y_pol = grp_sent.create_dataset("yPol", data=sent_y)
        pass
    with open(os.path.join(save_path, "val_fold" + str(fold) + ".hdf5"), "w") as fh5:
        pass
    with open(os.path.join(save_path, "test_fold" + str(fold) + ".hdf5"), "w") as fh5:
        pass


def get_all_symbols_data(datafile, data_path=None):  # data_path=RAW_MERGE_HDF5_DATASET_PATH
    """
    get all data symbols stored as dict
    :return symbols_data, {recv:{xPol:data, yPol:data}, sent:{xPol:data, yPol:data}}
    """
    f = h5py.File(datafile, 'r')
    symbols_data = {}
    for grp, _ in f.items():
        symbols_data[grp] = {}
        for ds, data in f.get(grp).items():
            symbols_data[grp][ds] = data
    return symbols_data


# Could rename this as get_input_data
def get_recv_symbols_data(datafile, grp='recv', data_path=None):  # data_path=RAW_MERGE_HDF5_DATASET_PATH
    """
    get all input data (recv symbols) as dict
    :return: recv_symbols_data{xPol: data, yPol: data}
    """
    recv_symbols_data = {}
    f = h5py.File(datafile, 'r')
    for pol in f[grp]:
        recv_symbols_data[pol] = f[grp][pol]
    return recv_symbols_data


# Could rename this as get_target_data
def get_sent_symbol_data(datafile, grp='sent', data_path=None):  # data_path=RAW_MERGE_HDF5_DATASET_PATH
    """
        get all desired output (sent symbols) as dict
        :return: sent_symbols_data{xPol: data, yPol: data}
        """
    sent_symbols_data = {}
    f = h5py.File(datafile, 'r')
    for pol in f[grp]:
        sent_symbols_data[pol] = f[grp][pol]
    return sent_symbols_data


# Split data from hdf5, try and get from hdf5 file
def split_data_train_val_test_from_hdf5(h5filename_pstr,
                              h5data_save_path=None,
                              raw_hdf5_data_path=None):
    # h5data_save_path=CROSS_VALIDATION_DATA_PATH,
    # raw_hdf5_data_path=RAW_MERGE_HDF5_DATASET_PATH
    h5filename = h5filename_pstr + '.hdf5'
    symbols_data = get_all_symbols_data(h5filename)
    for fold in range(1, NUM_CROSS_VAL_FOLDS+1):
        print("Splitting train-test fold", fold)
        # Get indexes
        subgrp = symbols_data[list(symbols_data)[0]]
        n_symbols = subgrp[list(subgrp)[0]].shape[0]
        train_indexs = np.argwhere(np.arange(n_symbols) % 5 != fold - 1).flatten()
        test_indexs = np.argwhere(np.arange(n_symbols) % 5 == fold - 1).flatten()
        # Must be rounded to multiple of 10 to ensure same sample with different offset
        # does not appear in both val and test
        val_indexs = test_indexs[0: int(len(test_indexs) / 20) * 10]
        test_indexs = test_indexs[int(len(test_indexs) / 20) * 10:]
        assert len(np.intersect1d(train_indexs, val_indexs)) == 0
        assert len(np.intersect1d(val_indexs, test_indexs)) == 0
        assert len(np.intersect1d(train_indexs, test_indexs)) == 0

        (
            train_input_symbols,
            val_input_symbols,
            test_input_symbols
        ) = split_numpy_arrays(list(symbols_data['recv'].values()), train_indexs, val_indexs, test_indexs)

        (
            train_target_symbols,
            val_target_symbols,
            test_target_symbols
        ) = split_numpy_arrays(list(symbols_data['sent'].values()), train_indexs, val_indexs, test_indexs)

        assert len(train_input_symbols) == len(train_target_symbols)
        assert len(val_input_symbols) == len(val_target_symbols)
        assert len(test_input_symbols) == len(test_target_symbols)

        save_numpy_data(
            os.path.join(CROSS_VALIDATION_DATA_INPUTS_PATH, h5filename_pstr),
            fold,
            train_input_symbols, val_input_symbols, test_input_symbols)

        save_numpy_data(
            os.path.join(CROSS_VALIDATION_DATA_TARGETS_PATH, h5filename_pstr),
            fold,
            train_target_symbols, val_target_symbols, test_target_symbols)
        pass


def split_data_train_val_test_all_from_hdf5(data_path=None):  # for all launch powers
    # data_path=RAW_MERGE_HDF5_DATASET_PATH
    os.chdir(data_path)
    h5data_save_path = CROSS_VALIDATION_DATA_PATH
    if not os.path.exists(h5data_save_path):
        os.makedirs(h5data_save_path)
    for plch in LAUNCH_POWER_RANGE_LIST:
        plch_str = str(plch) + '_Pdbm'
        # h5filename = plch_str + '.hdf5'
        # h5data_plch_save_path = os.path.join(h5data_save_path, plch_str)
        # if not os.path.exists(h5data_plch_save_path):
        #     os.makedirs(h5data_plch_save_path)
        split_data_train_val_test_from_hdf5(plch_str, h5data_save_path)


if __name__ == "__main__":
    # tmpfile = TMP_DATA_PATH + 'data.h5'
    # f = h5py.File(tmpfile, 'r')
    # path = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/torch/tmp'
    # os.chdir(path)
    # split_data_train_val_test('0_Pdbm', CROSS_VALIDATION_DATA_PATH)
    # split_data_train_val_test_all()

    # h5filename = 'test.hdf5'
    # f = h5py.File(h5filename, 'r')
    # a = f.get('recv').get('xPol').value
    # # get_all_symbols_data(h5filename)
    # get_recv_symbols_data(h5filename)
    pass
