# Internal
from preprocessing.preprocess_grp_data import create_dataset_from_hdf5

# External
import h5py
import numpy as np
# from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import time
import torch
from torch.utils.data import Dataset
import random
import os


class DataSymbolsDataset(Dataset):
    def __init__(self, inputs_path, targets_path, grp_targets_path=None,
                 flatten_inputs=False, flatten_targets=False, flatten_grp_targets=False):
        self.inputs = np.load(inputs_path)
        self.targets = np.load(targets_path)
        self.grp_targets = np.load(grp_targets_path) if grp_targets_path is not None else None
        self.flatten_inputs = flatten_inputs
        self.flatten_targets = flatten_targets
        self.flatten_grp_targets = flatten_grp_targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):  # For extracting inputs and targets from batch directly, see trainer.py
        batch_inputs = self.inputs[idx]
        batch_targets = self.targets[idx]
        if self.flatten_inputs:
            batch_inputs = batch_inputs.flatten()
        if self.flatten_targets:
            batch_targets = batch_targets.flatten()

        if self.grp_targets is not None:
            batch_grp_targets = self.grp_targets[idx]
            if self.flatten_grp_targets:
                batch_grp_targets = batch_grp_targets.flatten()
            return {"inputs": batch_inputs, "targets": batch_targets, "grp_targets": batch_grp_targets}
        else:
            return {"inputs": batch_inputs, "targets": batch_targets}


def get_data_symbols_dataset(
        inputs_path,
        targets_path,
        batch_size,
        shuffle,
        num_workers,
        grp_targets_path=None,
        phase=None,
        fold=None,
        flatten_inputs=False,
        flatten_targets=False,
        flatten_grp_targets=False
):
    # get dataset with split train/test/val datasets

    if fold:
        filename = phase + "_fold" + str(fold) + ".npy"
    else:
        filename = phase + ".npy"

    inputs_path = os.path.join(inputs_path, filename)
    targets_path = os.path.join(targets_path, filename)
    grp_targets_path = os.path.join(grp_targets_path, filename) if grp_targets_path is not None else None

    dataset = DataSymbolsDataset(
        inputs_path, targets_path, grp_targets_path, flatten_inputs, flatten_targets
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return dataset_loader


"""Ref version of data loader"""


# ============================== [Start] DataLoader ============================== #
def data_loader(input_data, target_data, batch_size, cuda):
    # received data symbols as input, sent data symbols as target
    """ data type -> np array after loaded from h5file """
    if cuda:
        input_tensor = torch.tensor(input_data, dtype=torch.double).cuda()  # input np array to tensor
        target_tensor = torch.tensor(target_data, dtype=torch.double).cuda()  # target np array to tensor

        tensor_dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)  # tensor to tensor_dataset
        dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
        # tensor_dataset to tensor_data_loader
    else:  # cpu
        pass

    return dataloader


def dataset_loader_hdf5(train_file, test_file, n_sym=41, guard_band=10 ** 3, n_splits=5, batch_size=1, seed=10,
                        cuda=True):
    """ from filename to create dataset, then to tensors and dataloader,
    including cross-validation splitting datasets, return train & valid & test datasets """
    # load np array data from h5 file -> split train & validation data -> tensor -> dataset -> dataloader
    # ------------------- load np array dataset ------------------- #
    train_data_recv, train_data_sent = create_dataset_from_hdf5(train_file, n_sym, guard_band)
    test_data_recv, test_data_sent = create_dataset_from_hdf5(test_file, n_sym, guard_band)

    # ------------------- add k-fold validation dataset for training data ------------------- #
    # SKFold needs input labels, use k-fold instead
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).
                  split(train_data_recv, train_data_sent))
    for i, (train_idx, valid_idx) in enumerate(splits):  # cross-validation with multiple splits, TBD
        pass

    # Using only one split of data for now
    idx = random.randrange(n_splits)
    train_idx, valid_idx = splits[idx][0], splits[idx][1]
    train_recv, train_sent = train_data_recv[train_idx.astype(int)], train_data_sent[train_idx.astype(int)]
    valid_recv, valid_sent = train_data_recv[valid_idx.astype(int)], train_data_sent[valid_idx.astype(int)]
    train_loader = data_loader(train_recv, train_sent, batch_size, cuda)
    valid_loader = data_loader(valid_recv, valid_sent, batch_size, cuda)

    # ------------------- create loader for test dataset ------------------- #
    test_loader = data_loader(test_data_recv, test_data_sent, batch_size, cuda)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    inputs_path = ""
    targets_path = ""
    dataset_args = {
        "inputs_path": inputs_path,
        # os.path.join(CROSS_VALIDATION_DATA_INPUTS_PATH, plch_str),
        "targets_path": targets_path,
        # os.path.join(CROSS_VALIDATION_DATA_TARGETS_PATH, plch_str),
        "cross_val": False,
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
        "flatten_inputs": False,
    }

    pass
    # print(y.shape)
    # print(alpha)
