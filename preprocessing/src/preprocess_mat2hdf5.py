"""Preprocess mat files and save as hdf5 files"""
# Internal
from global_config.global_config import (
    SYSTEM_CONFIGURATION,
    LAUNCH_POWER_RANGE_LIST,
)
from preprocessing.config.preprocessing_config import (
    TRAIN_MAT_DATASET_PATH,
    TEST_MAT_DATASET_PATH,
    TRAIN_HDF5_DATASET_PATH,
    TEST_HDF5_DATASET_PATH
)

from preprocessing.utils.preprocessing_utils import cplx2real

# External
import os
from scipy.io import loadmat
import h5py


def mat2hdf5(matfilename, h5filename, h5datapath):  # h5datapath=RAW_MERGE_HDF5_DATASET_PATH
    mat = loadmat(matfilename)
    mat_data = [mat['X_in'], mat['X_des'], mat['Y_in'], mat['Y_des']]
    recv_x, sent_x, recv_y, sent_y = cplx2real(mat_data)
    os.chdir(h5datapath)
    f = h5py.File(h5filename, "w")
    grp_recv = f.create_group('recv')
    grp_recv.create_dataset("xPol", data=recv_x)
    grp_recv.create_dataset("yPol", data=recv_y)
    # recv_x_pol = grp_recv.create_dataset_from_hdf5("xPol")
    # recv_y_pol = grp_recv.create_dataset_from_hdf5("yPol")

    grp_sent = f.create_group('sent')
    grp_sent.create_dataset("xPol", data=sent_x)
    grp_sent.create_dataset("yPol", data=sent_y)
    # sent_x_pol = grp_sent.create_dataset_from_hdf5("xPol")
    # sent_y_pol = grp_sent.create_dataset_from_hdf5("yPol")


def create_hdf5_datasets(matdatapath, h5datapath):
    # matdatapath=RAW_MERGE_MAT_DATASET_PATH, h5datapath=RAW_MERGE_HDF5_DATASET_PATH
    if not os.path.exists(h5datapath):
        os.makedirs(h5datapath)
    file_list = os.listdir(matdatapath)
    for i, _ in enumerate(file_list):
        for plch in LAUNCH_POWER_RANGE_LIST:
            plch_str = str(plch) + '_Pdbm'

            if file_list[i].__contains__(plch_str):
                mat_file_name = os.path.join(matdatapath, file_list[i])
                h5_file_name = plch_str + '.hdf5'
                mat2hdf5(mat_file_name, h5_file_name, h5datapath)
                break





if __name__ == "__main__":

    create_hdf5_datasets(TRAIN_MAT_DATASET_PATH, TRAIN_HDF5_DATASET_PATH)
    create_hdf5_datasets(TEST_MAT_DATASET_PATH, TEST_HDF5_DATASET_PATH)
    '''
    path = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/torch/tmp'
    os.chdir(path)
    h5filename = '2_Pdbm.hdf5'
    f = h5py.File(h5filename, 'r')
    a = f.get('recv')
    for x, y in f.items():
        print(x)
        print(y)
    print(a.keys())
    for x, y in a.items():
        print(x)
        print(y)
    for _ in f:
        for __ in f[_]:
            print(__)
    '''

    '''
    # create_hdf5_datasets()
    path = '/home/liu2_y@WMGDS.WMG.WARWICK.AC.UK/PycharmProjects/torch/tmp'
    # name = 'test.mat'
    # matfilename = path + '/' + name
    os.chdir(path)
    h5filename = 'test.hdf5'
    f = h5py.File(h5filename, 'r')
    a = f.get('recv')
    for x, y in f.items():
        print(x)
        print(y)
    print(a.keys())
    for x, y in a.items():
        print(x)
        print(y)
    for _ in f:
        for __ in f[_]:
            print(__)
    '''
    # pass
    # mat = loadmat(matfilename)
    # dict = [mat['X_in'], mat['X_des'], mat['Y_in'], mat['Y_des']]
    # a, b, c, d = cplx2real(dict)
    # mat2hdf5(matfilename, h5filename, path)


# -- ARCHIVE -- #
# -------------------------------------------------------------------------------------- #
# merge mats from train and test folders containing data with same Plchs, currently not used
# def merge_mat():  # put train and test mats of same plch in one mat list
#     dir_list = os.listdir(RAW_DATASET_PATH)
#     # Merging train and test .mat data in one based on plch for later split
#     for plch in LAUNCH_POWER_RANGE_LIST:
#         plch_str = str(plch) + '_Pdbm'
#         mats = []
#         for subdir in dir_list:
#             subdir = os.path.join(RAW_DATASET_PATH, subdir)
#             file_list = os.listdir(subdir)
#
#             for i, _ in enumerate(file_list):
#                 if file_list[i].__contains__(plch_str):
#                     mat_file_name = file_list[i]
#                     break
#             mat = loadmat(os.path.join(subdir, mat_file_name))
#             mats.append(mat)