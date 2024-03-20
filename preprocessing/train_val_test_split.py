# Internal
from global_config.global_config import (
    NUM_CROSS_VAL_FOLDS,
    LAUNCH_POWER_RANGE_LIST,
    ALL_DATA_PATH,
    ALL_INPUTS_DATA_PATH,
    ALL_TARGETS_DATA_PATH,
    ALL_GRP_TARGETS_DATA_PATH,
    CROSS_VAL_DATA_PATH,
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


def save_numpy_data(save_path, fold, train_data, val_data, test_data):
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


for plch in LAUNCH_POWER_RANGE_LIST:
    plch_str = str(plch) + '_Pdbm'
    npyfile = plch_str + '.npy'
    inputs = np.load(os.path.join(ALL_INPUTS_DATA_PATH, "inputs_" + npyfile))
    targets = np.load(os.path.join(ALL_TARGETS_DATA_PATH, "targets_" + npyfile))
    grp_targets = np.load(os.path.join(ALL_GRP_TARGETS_DATA_PATH, "grp_targets_" + npyfile))

    for fold in range(1, NUM_CROSS_VAL_FOLDS+1):
        print("Splitting train-test fold", fold, "for plch", plch_str)
        # Get indexes
        data_symbols_len = inputs.__len__()

        train_indexs = np.argwhere(np.arange(data_symbols_len) % 5 != fold - 1).flatten()
        test_indexs = np.argwhere(np.arange(data_symbols_len) % 5 == fold - 1).flatten()
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
        ) = split_numpy_array(inputs, train_indexs, val_indexs, test_indexs)

        (
            train_target_symbols,
            val_target_symbols,
            test_target_symbols
        ) = split_numpy_array(targets, train_indexs, val_indexs, test_indexs)

        (
            train_grp_target_symbols,
            val_grp_target_symbols,
            test_grp_target_symbols
        ) = split_numpy_array(grp_targets, train_indexs, val_indexs, test_indexs)

        assert len(train_input_symbols) == len(train_target_symbols) == len(train_grp_target_symbols)
        assert len(val_input_symbols) == len(val_target_symbols) == len(val_grp_target_symbols)
        assert len(test_input_symbols) == len(test_target_symbols) == len(test_grp_target_symbols)

        save_numpy_data(
            os.path.join(CROSS_VAL_INPUTS_DATA_PATH, plch_str),
            fold,
            train_input_symbols, val_input_symbols, test_input_symbols)

        save_numpy_data(
            os.path.join(CROSS_VAL_TARGETS_DATA_PATH, plch_str),
            fold,
            train_target_symbols, val_target_symbols, test_target_symbols)

        save_numpy_data(
            os.path.join(CROSS_VAL_GRP_TARGETS_DATA_PATH, plch_str),
            fold,
            train_grp_target_symbols, val_grp_target_symbols, test_grp_target_symbols)





