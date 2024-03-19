"""Transformer encoder-only"""
# Internal
from experiments.datasets import get_data_symbols_dataset
from experiments.trainer import train_model, test_model
from global_config.global_config import (
    # LAUNCH_POWER_RANGE_LIST,  # from parser
    N_ADJACENT_SYMBOLS,
    N_FEATURES,
    NUM_CROSS_VAL_FOLDS,
    CROSS_VAL_DATA_PATH,
    CROSS_VAL_INPUTS_DATA_PATH,
    CROSS_VAL_TARGETS_DATA_PATH,
    CROSS_VAL_GRP_TARGETS_DATA_PATH,
    OUTPUT_COLLECTION_PATH
)

from experiments.models import TransformerEncoderMLP

from utils.utils import check_makedir, model_summary, training_output
from utils.parsers import get_arg_array, TransformerEncoderParser
from utils.helper import get_ref_ber_q_from_npy

# External
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import os

# to free unoccupied cached memory
torch.cuda.empty_cache()
# ########## ARGUMENT PARSER ########## #
args, args_dir_str = TransformerEncoderParser().parsers()
DEVICE = torch.device("cuda:" + args.gpu_id)
# NUM_LAYERS = int(args.n_layers)
NUM_LAYERS = 1

torch.cuda.set_device(DEVICE)
# ########## SYSTEM CONFIG ########## #
# Launch_power preprocess
LAUNCH_POWER_RANGE = get_arg_array(args.plch)

# ########## NETWORK CONFIG ########## #
# DEVICE = torch.device("cuda:0")
N_FEATURES = 2  # To be deleted
INPUT_SIZE = N_FEATURES
OUTPUT_SIZE = 2
SEQ_LEN = N_ADJACENT_SYMBOLS
LEARNING_RATE = 1e-3
BATCH_SIZE = 4331
NUM_WORKERS = 4
WEIGHT_DECAY = 0
DEBUG_MODE = bool(args.debug == 't')
NUM_EPOCHS = 3 if DEBUG_MODE is True else 1000
MODEL_SAVE_PATH = os.path.join(OUTPUT_COLLECTION_PATH, "transformer_linear_encoder")

network_args = {"device": DEVICE, "embed": False, "embed_size": 32, "input_size": INPUT_SIZE, "num_layers": NUM_LAYERS,
                 "heads": 1, "forward_expansion": 4, "dropout": 0, "encoder_type": "linear", "mlp_depth": 2,
                "mlp_widths":[100,100], "output_size": OUTPUT_SIZE, "seq_len": SEQ_LEN}

for plch in LAUNCH_POWER_RANGE:
    plch_str = str(plch) + "_Pdbm"

    outputdir = os.path.join(MODEL_SAVE_PATH, args_dir_str, plch_str)
    check_makedir(outputdir)

    for fold in range(1, NUM_CROSS_VAL_FOLDS + 1):

        # ########## SET UP DATASET ########## #
        train_dataset_args = {
            "inputs_path": os.path.join(CROSS_VAL_INPUTS_DATA_PATH, plch_str),
            "targets_path": os.path.join(CROSS_VAL_TARGETS_DATA_PATH, plch_str),
            "grp_targets_path": os.path.join(CROSS_VAL_GRP_TARGETS_DATA_PATH, plch_str),
            "fold": fold,
            "phase": "train",
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": NUM_WORKERS,
            "flatten_inputs": False,
        }

        val_dataset_args = {
            "inputs_path": os.path.join(CROSS_VAL_INPUTS_DATA_PATH, plch_str),
            "targets_path": os.path.join(CROSS_VAL_TARGETS_DATA_PATH, plch_str),
            "grp_targets_path": os.path.join(CROSS_VAL_GRP_TARGETS_DATA_PATH, plch_str),
            "fold": fold,
            "phase": "val",
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "flatten_inputs": False,
        }

        test_dataset_args = {
            "inputs_path": os.path.join(CROSS_VAL_INPUTS_DATA_PATH, plch_str),
            "targets_path": os.path.join(CROSS_VAL_TARGETS_DATA_PATH, plch_str),
            "grp_targets_path": os.path.join(CROSS_VAL_GRP_TARGETS_DATA_PATH, plch_str),
            "fold": fold,
            "phase": "test",
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "num_workers": NUM_WORKERS,
            "flatten_inputs": False,  # check, previously True, not working on models
        }

        train_loader = get_data_symbols_dataset(**train_dataset_args)
        val_loader = get_data_symbols_dataset(**val_dataset_args)
        test_loader = get_data_symbols_dataset(**test_dataset_args)

        # ########## SET UP MODEL ########## #
        model = TransformerEncoderMLP(**network_args).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_function = nn.MSELoss()
        training_loss = []  # For visualising training loss
        model_summary(model=model, input_size=(BATCH_SIZE, SEQ_LEN, N_FEATURES), path=outputdir)  # print model summary

        # ########## TRAIN AND EVALUATE ########## #
        best_ber = 1

        # ########## Reference Q-factor ########## #
        ref_ber, ref_q = get_ref_ber_q_from_npy(os.path.join(CROSS_VAL_DATA_PATH, "Plch_BER_Q.npy"), plch)

        for epoch in range(NUM_EPOCHS):
            training_output(output={'epoch_num': epoch}, path=outputdir)  # print current epoch number

            training_output(output={'phase': 'reference', 'ber': ref_ber, 'q': ref_q},
                            path=outputdir)  # print reference BER and Q

            trainer_args = {
                "model": model,
                "device": DEVICE,
                "train_loader": train_loader,
                "optimizer": optimizer,
                "loss_function": loss_function,
                "debug_mode": DEBUG_MODE,
                "attention": False,   #  Change to true afterwards
            }

            loss, ber, q = train_model(**trainer_args)
            training_loss.append(loss)  # For appending loss from each epoch

            training_output(output={'phase': 'train', 'loss': loss, 'ber': ber, 'q': q},
                            path=outputdir)  # print train output

            val_args = {
                "model": model,
                "device": DEVICE,
                "test_loader": val_loader,
                "loss_function": loss_function,
                "debug_mode": DEBUG_MODE,
                "attention": False,   #  Change to true afterwards
            }

            # loss, ber, q, _ = test_model(**val_args)
            loss, ber, q = test_model(**val_args)

            training_output(output={'phase': 'validation', 'loss': loss, 'ber': ber, 'q': q},
                            path=outputdir)  # print validation output

            if ber < best_ber:
                idx = epoch
                best_ber = ber
                best_model = copy.deepcopy(model)
                model_save_name = "fold_" + str(fold) + ".weights"
                torch.save(best_model.state_dict(), os.path.join(outputdir, model_save_name))

            if (epoch - idx) >= 150:
                training_loss = np.asarray(training_loss)
                training_loss_save_path = os.path.join(MODEL_SAVE_PATH, args_dir_str, plch_str)
                check_makedir(training_loss_save_path)
                np.save(os.path.join(training_loss_save_path, "training_loss.npy"), training_loss)
                break

            test_args = {
                "model": best_model,
                "device": DEVICE,
                "test_loader": test_loader,
                "loss_function": loss_function,
                "debug_mode": False,
                "attention": False,   #  Change to true afterwards
                "fold_num": fold,
                "predictions_save_path": os.path.join(MODEL_SAVE_PATH, args_dir_str, plch_str)
            }

            # loss, ber, q, att_weights = test_model(**test_args)
            loss, ber, q = test_model(**test_args)
            training_output(output={'phase': 'test', 'loss': loss, 'ber': ber, 'q': q},
                            path=outputdir)  # print test output

            # if epoch % 3 == 0:
            #     # save attention weights of the last output
            #     att_weights_filename = 'epoch_' + str(epoch) + '.npy'
            #     att_weights_save_path = os.path.join(MODEL_SAVE_PATH, args_dir_str, plch_str,  "attention_weights")
            #     check_makedir(att_weights_save_path)
            #     np.save(os.path.join(att_weights_save_path, att_weights_filename), att_weights[-1, :, 0])

            if epoch == (NUM_EPOCHS - 1):
                training_loss = np.asarray(training_loss)
                training_loss_save_path = os.path.join(MODEL_SAVE_PATH, args_dir_str, plch_str)
                check_makedir(training_loss_save_path)
                np.save(os.path.join(training_loss_save_path, "training_loss.npy"), training_loss)
