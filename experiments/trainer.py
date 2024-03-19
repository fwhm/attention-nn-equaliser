# Internal
from utils.ber_calc import qam_ber_gray, ber2q
from utils.helper import real2complex
from global_config.global_config import (
    MODULATION_ORDER,
)

# External
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
# from sklearn.metrics import average_precision_score
import os


def grad_clipping(model, theta):
    """Clip the gradient."""
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.GRU:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


def get_n_taps_x(x, n_taps, input_size):
    mid_idx = x.shape[1]//2
    # return x[:, mid_idx-n_taps:mid_idx+n_taps+1, :]
    return x[:, mid_idx-n_taps:mid_idx+n_taps+1, 0:input_size]


def train_model(model, device, train_loader, optimizer, loss_function, debug_mode, compute_ber_q=True, attention=False, compute_ap=False):
    # train model with split train/test/val datasets
    model.train()
    loss = 0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = get_n_taps_x(batch["inputs"], model.n_taps, model.input_size).to(device)
        targets = batch["targets"].to(device)
        inputs = inputs.float()
        targets = targets.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        # Forward
        if attention:
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        all_predictions[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    total_loss /= len(train_loader)

    if compute_ber_q:
        ber = qam_ber_gray(real2complex(all_predictions), real2complex(all_targets), MODULATION_ORDER)
        q = ber2q(ber)
    else:
        ber = 1
        pass
    '''
    if compute_ap:
        ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
    else:
        ap = 0
    '''

    return total_loss, ber, q


def test_model(
        model,
        device,
        test_loader,
        loss_function,
        debug_mode,
        fold_num=None,
        predictions_save_path=None,
        predictions_save_shape=None,
        compute_ber_q=True,
        compute_ap=False,
        variable_batch_size=False,
        attention=False
):
    # test/evaluate model with split train/test/val datasets
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 10:
                break

            # Read data
            # inputs = batch["inputs"].to(device)
            inputs = get_n_taps_x(batch["inputs"], model.n_taps, model.input_size).to(device)
            targets = batch["targets"].to(device)
            inputs = inputs.float()
            targets = targets.float()
            # inputs = torch.squeeze(inputs).float()
            # targets = torch.squeeze(targets).float()

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            # Forward
            if attention:
                outputs, att_weights = model(inputs)
                att_weights = att_weights.cpu().numpy()  # record att_weights
            else:
                outputs = model(inputs)

            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)

        if compute_ber_q:
            ber = qam_ber_gray(real2complex(all_predictions), real2complex(all_targets), MODULATION_ORDER)
            q = ber2q(ber)
        else:
            ber = 1

        '''
        if compute_ap:
            ap = average_precision_score(all_targets.flatten(), all_predictions.flatten())
        else:
            ap = 0
        '''

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        # Save predictions
        np.save(os.path.join(predictions_save_path, "pred_test_fold" + str(fold_num) + ".npy"), all_predictions)
    if attention:
        return loss, ber, q, att_weights
    else:
        return loss, ber, q

def train_encoder_decoder():
    pass


# def train_seq2seq(net, data_iter, lr, num_epochs, device):
def train_seq2seq(model, device, train_loader, optimizer, loss_function, debug_mode, compute_ber_q=True):
    """Train a model for sequence to sequence."""
    model.train()
    loss=0
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if debug_mode and batch_idx > 10:
            break

        # Read data
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        dec_inputs = batch["grp_targets"].to(device)  # dataset grp_targets is used for decoder_inputs
        inputs = inputs.float()
        targets = targets.float()
        dec_inputs = dec_inputs.float()

        if batch_idx == 0:
            batch_size = targets.shape[0]
            all_predictions = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])
            all_targets = np.zeros((batch_size * len(train_loader),) + targets.shape[1:])

        # Forward
        outputs, _ = model(inputs, dec_inputs)  # dec_inputs is for teaching force
        # the original *args input of the model valid_len is not used in the current one
        loss = loss_function(outputs[:, int((outputs.shape[1]-1)/2), 0:2], targets)
        # loss = loss_function(outputs, dec_inputs)

        optimizer.zero_grad()
        loss.sum().backward()
        grad_clipping(model, 1)
        optimizer.step()
        total_loss += loss.item()

        all_targets[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
        # all_predictions[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = outputs.cpu().detach()
        all_predictions[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = \
            outputs[:, int((outputs.shape[1]-1)/2), 0:2].cpu().detach()

    final_batch_size_difference = batch_size - targets.shape[0]
    if final_batch_size_difference != 0:
        all_targets = all_targets[0:-final_batch_size_difference]
        all_predictions = all_predictions[0:-final_batch_size_difference]

    total_loss /= len(train_loader)

    if compute_ber_q:
        ber = qam_ber_gray(real2complex(all_predictions), real2complex(all_targets), MODULATION_ORDER)
        q = ber2q(ber)
    else:
        ber = 1

    return total_loss, ber, q


def test_seq2seq(
        model,
        device,
        test_loader,
        loss_function,
        debug_mode,
        fold_num=None,
        predictions_save_path=None,
        predictions_save_shape=None,
        compute_ber_q=True,
        compute_ap=False,
        variable_batch_size=False,
):
    # test/evaluate model with split train/test/val datasets
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 0:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            dec_inputs = batch["grp_targets"].to(device)  # dataset grp_targets is used for decoder_inputs
            inputs = inputs.float()
            targets = targets.float()
            dec_inputs = dec_inputs.float()
            # TBD Comment out the dec_inputs

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            # seq_prediction = torch.empty((batch_size, inputs.shape[1], targets.shape[-1]), device=device)
            if batch_idx == len(test_loader) - 1:  # For creating seq_prediction the same size as last batch
                last_batch_size = targets.shape[0]
                seq_prediction = torch.empty((last_batch_size, inputs.shape[1], inputs.shape[-1]), device=device)
            else:
                seq_prediction = torch.empty((batch_size, inputs.shape[1], inputs.shape[-1]), device=device)
            # seq_prediction is used inbetween encoder-decoder, so move to device

            # Forward
            for step_idx in np.arange(inputs.shape[1]):
                if step_idx == 0:  # Step 0 for init_state of the decoder
                    enc_outputs = model.encoder(inputs)  # using all input sequence for decoder init_state
                    dec_state = model.decoder.init_state(enc_outputs)
                    break  # break after init_state as step_idx becomes larger than zero

            # predict whole sequence altogether
            outputs, dec_state = model.decoder(
                    inputs, dec_state)

            # only regarding mid-idx of seq_prediction as outputs.
            # Then the whole sequence predicting is not making sense anymore
            # save for later use, e.g. use whole sequence to calculate loss
            #

            loss += loss_function(outputs[:, int((outputs.shape[1]-1)/2), 0:2], targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]
                ] = outputs[:, int((outputs.shape[1]-1)/2), 0:2].cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)

        if compute_ber_q:
            ber = qam_ber_gray(real2complex(all_predictions), real2complex(all_targets), MODULATION_ORDER)
            q = ber2q(ber)
        else:
            ber = 1

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        # Save predictions
        np.save(os.path.join(predictions_save_path, "pred_test_fold" + str(fold_num) + ".npy"), all_predictions)

    return loss, ber, q


def test_seq2seq_step(
        model,
        device,
        test_loader,
        loss_function,
        debug_mode,
        fold_num=None,
        predictions_save_path=None,
        predictions_save_shape=None,
        compute_ber_q=True,
        compute_ap=False,
        variable_batch_size=False,
):
    # test/evaluate model with split train/test/val datasets
    model.eval()
    loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if debug_mode and batch_idx > 0:
                break

            # Read data
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            dec_inputs = batch["grp_targets"].to(device)  # dataset grp_targets is used for decoder_inputs
            inputs = inputs.float()
            targets = targets.float()
            dec_inputs = dec_inputs.float()
            # TBD Comment out the dec_inputs

            if batch_idx == 0:
                batch_size = targets.shape[0]
                if variable_batch_size:
                    all_predictions = np.empty((0,) + targets.shape[1:])
                    all_targets = np.empty((0,) + targets.shape[1:])
                else:
                    all_predictions = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])
                    all_targets = np.zeros((batch_size * len(test_loader),) + targets.shape[1:])

            # seq_prediction = torch.empty((batch_size, inputs.shape[1], targets.shape[-1]), device=device)
            if batch_idx == len(test_loader)-1:  # For creating seq_prediction the same size as last batch
                last_batch_size = targets.shape[0];
                seq_prediction = torch.empty((last_batch_size, inputs.shape[1], inputs.shape[-1]), device=device)
            else:
                seq_prediction = torch.empty((batch_size, inputs.shape[1], inputs.shape[-1]), device=device)
            # seq_prediction is used inbetween encoder-decoder, so move to device

            # Forward
            for step_idx in np.arange(inputs.shape[1]):
                if step_idx == 0:  # Step 0 for init_state of the decoder
                    enc_outputs = model.encoder(inputs)  # using all input sequence for decoder init_state
                    dec_state = model.decoder.init_state(enc_outputs)
                    '''
                    # unsqueeze for expand_dim after slicing, attempted, but TypeError:
                    # can't convert cuda:0 device type tensor to numpy.
                    # Use Tensor.cpu() to copy the tensor to host memory first.
                    seq_prediction[:, step_idx, :], dec_state = model.decoder(
                        torch.unsqueeze(inputs[:, step_idx, :], 1), dec_state)
                    '''
                    # Simple solution: keep tensor dim by using a one-element slice
                    seq_prediction[:, step_idx:step_idx+1, :], dec_state = model.decoder(
                        inputs[:, step_idx:step_idx+1, :], dec_state)
                    continue  # continue as step_idx becomes larger than zero
                # use the predicted symbol for next decoder's input
                seq_prediction[:, step_idx:step_idx+1, :], dec_state = model.decoder(
                    seq_prediction[:, step_idx-1:step_idx, :], dec_state)

            # only regarding mid-idx of seq_prediction as outputs.
            # Then the whole sequence predicting is not making sense anymore
            # save for later use, e.g. use whole sequence to calculate loss
            #
            outputs = seq_prediction[:, int((targets.shape[1]-1)/2), 0:2]
            loss += loss_function(outputs, targets)

            if variable_batch_size:
                all_targets = np.append(all_targets, targets.cpu())
                all_predictions = np.append(all_predictions, outputs.cpu().detach())
            else:
                all_targets[batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]] = targets.cpu()
                all_predictions[
                    batch_size * batch_idx: (batch_size * batch_idx) + targets.shape[0]
                ] = outputs.cpu().detach()

        if not variable_batch_size:
            final_batch_size_difference = batch_size - targets.shape[0]
            if final_batch_size_difference != 0:
                all_targets = all_targets[0:-final_batch_size_difference]
                all_predictions = all_predictions[0:-final_batch_size_difference]

        loss /= len(test_loader)

        if compute_ber_q:
            ber = qam_ber_gray(real2complex(all_predictions), real2complex(all_targets), MODULATION_ORDER)
            q = ber2q(ber)
        else:
            ber = 1

    if predictions_save_path is not None:
        if not os.path.exists(predictions_save_path):
            os.makedirs(predictions_save_path)
        if predictions_save_shape is not None:
            all_predictions = all_predictions.reshape((-1,) + predictions_save_shape)
        # Save predictions
        np.save(os.path.join(predictions_save_path, "pred_test_fold" + str(fold_num) + ".npy"), all_predictions)
    return loss, ber, q


