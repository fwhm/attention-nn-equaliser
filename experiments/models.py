# https://github.com/olly-styles/Trajectory-Tensors/blob/main/experiments/models.py
# Define NN modules
# Internal
from experiments.network_components import (
    Flatten,
    SimpAttLayer,
    SimpAttInputLayer,
    TrimRnnOutTapsLayer,
    SelfAttentionLayer
)
from model.attentions import *
from experiments.transformer import *

# External
import torch
import torch.nn as nn
import torch.nn.init as init


class Recurrent(nn.Module):
    def __init__(self, device, rnn_type, n_hiddens=4, n_layers=1, bidirectional=False, unpack_bidir=False,
                 input_size=4, output_size=2, seq_len=41, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        self.device = device
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_taps = self.seq_len//2
        self.input_size = input_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.unpack_dir = unpack_bidir
        self.rnn_type = rnn_type
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                               batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                              batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                              batch_first=True, bidirectional=self.bidirectional)
        else:
            pass
        self.flatten = Flatten()
        if self.bidirectional:
            self.dense = nn.Linear(self.seq_len * 2 * self.n_hiddens, self.output_size)

        else:
            self.dense = nn.Linear(self.seq_len * self.n_hiddens, self.output_size)

    def get_n_taps_x(self, x):
        mid_idx = x.shape[1]//2
        return x[:, mid_idx-self.n_taps:mid_idx+self.n_taps+1, :]

    def forward(self, x, *args):
        self.rnn.flatten_parameters()
        x = self.get_n_taps_x(x)  # take 2 * n_taps + 1 symbols
        out, _ = self.rnn(x)  # LSTM output: tuple(output,(h_n, c_n)), using "_" to ignore the second output
        if self.unpack_dir:
            out = torch.cat((out[:, :, :self.n_hiddens], out[:, :, self.n_hiddens:]), dim=1)
        out = self.flatten(out)  # reshape for flattening according to batch_size => out = out.reshape(x.shape[0], -1)
        out = self.dense(out)
        return out


class PartialBiRNN(Recurrent):
    """Partial BiRNN without trimming RNN output"""
    def __init__(self, fw_id, bw_id, trim, **kwargs):
        super(PartialBiRNN, self).__init__(**kwargs)
        self.fw_id = fw_id
        self.bw_id = bw_id
        if self.rnn_type == "LSTM":
            self.rnn_fw = nn.LSTM(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                  batch_first=True, bidirectional=False)
            self.rnn_bw = nn.LSTM(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                  batch_first=True, bidirectional=False)
        elif self.rnn_type == "GRU":
            self.rnn_fw = nn.GRU(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                 batch_first=True, bidirectional=False)
            self.rnn_bw = nn.GRU(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                 batch_first=True, bidirectional=False)
        elif self.rnn_type == "RNN":
            self.rnn_fw = nn.RNN(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                 batch_first=True, bidirectional=False)
            self.rnn_bw = nn.RNN(input_size=self.input_size, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                                 batch_first=True, bidirectional=False)
        else:
            pass
        self.fw_seq_len = self.n_taps + self.fw_id + 1
        self.bw_seq_len = self.n_taps - self.bw_id + 1
        self.trim_len = np.abs(self.fw_id - self.bw_id + 1)
        self.trim = trim
        if self.trim:
            self.dense = nn.Linear(2 * self.trim_len * self.n_hiddens, self.output_size)
        else:
            self.dense = nn.Linear((self.fw_seq_len + self.bw_seq_len) * self.n_hiddens, self.output_size)

    def get_fw_bw_x(self, fw_bw_type, x):
        mid_idx = x.shape[1] // 2

        if fw_bw_type == 'fw':
            x = x[:, :mid_idx+self.fw_id+1, :]
        elif fw_bw_type == 'bw':
            x = x[:, mid_idx+self.bw_id:, :]
            # flip along sequence axis for backward RNN input
            x = torch.flip(x, [1])
        else:
            raise ValueError("Must be either forward or backward!")

        return x

    def get_trimmed_rnn_out(self, x):
        """For both fw and bw, ----  fw -3 -2 -1 0 1 2 3 bw -3 -2 -1 0 1 2 3 """
        mid_idx = x.shape[1]//2  # rnn out length
        return x[:, mid_idx-1-self.trim_len:mid_idx+self.trim_len-1, :]

    def forward(self, x, *args):
        self.rnn_fw.flatten_parameters()
        self.rnn_bw.flatten_parameters()
        x_fw = self.get_fw_bw_x('fw', x)
        x_bw = self.get_fw_bw_x('bw', x)
        fw_rnn_out, _ = self.rnn_fw(x_fw)
        bw_rnn_out = torch.flip(self.rnn_bw(x_bw)[0], [1])  # flip back the backward rnn output
        rnn_out = torch.cat((fw_rnn_out, bw_rnn_out), dim=1)  # concat forward and backward rnn outputs
        if self.trim:
            rnn_out = self.get_trimmed_rnn_out(rnn_out)
        flatten_out = self.flatten(rnn_out)
        out = self.dense(flatten_out)
        return out


class PartialBiRNNExtraTrim(Recurrent):
    """Partial BiRNN with trimming RNN output"""
    pass


class InOutTapRnn(Recurrent):
    def __init__(self, in_tap_ids, out_tap_ids, **kwargs):
        super(InOutTapRnn, self).__init__(**kwargs)
        self.in_tap_ids = in_tap_ids
        self.out_tap_ids = out_tap_ids
        if self.bidirectional:
            self.dense = nn.Linear(len(self.out_tap_ids) * 2 * self.n_hiddens, self.output_size)
        else:
            self.dense = nn.Linear(len(self.out_tap_ids) * self.n_hiddens, self.output_size)

    def get_tap_ids_x(self, x, ids, in_out_tap='in'):
        if in_out_tap == 'in':
            mid_idx = self.seq_len//2
            return x[:, mid_idx+ids, :]
        elif in_out_tap == 'out':
            ids_diff = self.out_tap_ids[0] - self.in_tap_ids[0]
            return x[:, ids_diff+(self.out_tap_ids-self.out_tap_ids[0]), :]
        else:
            raise ValueError("tap_ids should have the type of in or out")

    def forward(self, x, *args):
        self.rnn.flatten_parameters()
        x = self.get_tap_ids_x(x, self.in_tap_ids, 'in')
        rnn_out, _ = self.rnn(x)
        trim_out = self.get_tap_ids_x(rnn_out, self.out_tap_ids, 'out')
        flatten_out = self.flatten(trim_out)
        out = self.dense(flatten_out)
        return out


class ConvRnn(InOutTapRnn):
    """CNN+BiLSTM"""
    def __init__(self, n_filters, kernel_size, stride=1, padding=0, dilation=1, **kwargs):  # n_filters is Cout
        super(ConvRnn, self).__init__(**kwargs)
        self.n_filters, self.kernel_size, self.stride, self.padding, self.dilation = \
            n_filters, kernel_size, stride, padding, dilation  # CNN params
        self.cnn_Lin = 2 * self.n_taps + 1  # Lin the same as seq_len
        self.cnn_Lout = int((self.cnn_Lin + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                            / self.stride + 1)
        self.cnn = nn.Conv1d(in_channels=self.input_size, out_channels=self.n_filters, kernel_size=self.kernel_size,
                             stride=self.stride, padding=self.padding, dilation=self.dilation)

        self.leakyRelu = nn.LeakyReLU()
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.n_filters, hidden_size=self.n_hiddens, num_layers=self.n_layers,
                               batch_first=True, bidirectional=self.bidirectional)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(self.n_filters, self.n_hiddens, self.n_layers, self.bidirectional, batch_first=True)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(self.n_filters, self.n_hiddens, self.n_layers, self.bidirectional, batch_first=True)
        else:
            pass
        if self.bidirectional:
            self.dense = nn.Linear(self.cnn_Lout * 2 * self.n_hiddens, self.output_size)
        else:
            self.dense = nn.Linear(self.cnn_Lout * self.n_hiddens, self.output_size)

    def forward(self, x, *args):
        x = x.permute(0, 2, 1)
        conv_out = self.cnn(x)
        lr_out = self.leakyRelu(conv_out)
        rnn_in = lr_out.permute(0, 2, 1)
        rnn_out, _ = self.rnn(rnn_in)
        flatten_out = self.flatten(rnn_out)
        out = self.dense(flatten_out)
        return out


class TrimRnnOut(Recurrent):
    def __init__(self, trim_tap=0, **kwargs):
        super(TrimRnnOut, self).__init__(**kwargs)
        self.trim_tap = trim_tap
        self.trim_len = 2 * self.trim_tap + 1
        self.trim_rnn_out = TrimRnnOutTapsLayer(self.trim_tap)
        self.dense = nn.Linear(self.trim_len * 2 * self.n_hiddens, self.output_size) if self.bidirectional \
            else nn.Linear(self.trim_len * self.n_hiddens, self.output_size)

    def forward(self, x, *args):
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(x)
        trim_out = self.trim_rnn_out(rnn_out)
        flatten_out = self.flatten(trim_out)
        out = self.dense(flatten_out)
        return out


class AttentionRecurrent(Recurrent):
    def __init__(self, return_sequence=True, **kwargs):
        super(AttentionRecurrent, self).__init__(**kwargs)
        self.return_sequence = return_sequence
        self._attention_weights = None

    @property
    def attention_weights(self):
        return self._attention_weights


class SimpAttLayerRecurrent(AttentionRecurrent):
    def __init__(self, return_sequence, **kwargs):
        super(SimpAttLayerRecurrent, self).__init__(**kwargs)
        self.att = SimpAttLayer(self.n_hiddens, self.seq_len, self.bidirectional, self.unpack_dir, return_sequence)

    def forward(self, x, *args):
        self.rnn.flatten_parameters()
        rnn_out, states = self.rnn(x)
        if self.unpack_dir:
            rnn_out = torch.cat((rnn_out[:, :, :self.n_hiddens], rnn_out[:, :, self.n_hiddens:]), dim=1)

        att_out, self._attention_weights = self.att(rnn_out)
        flatten = self.flatten(att_out)
        out = self.dense(flatten)
        return out, self._attention_weights


class SimpAttInput(AttentionRecurrent):
    """Simple attention before RNN layer, on the input sequence solely"""
    def __init__(self, return_sequence, **kwargs):
        super(SimpAttInput, self).__init__(**kwargs)
        self.att = SimpAttInputLayer(self.input_size, self.seq_len, return_sequence)

    def forward(self, x, *args):
        att_out, self._attention_weights = self.att(x)
        self.rnn.flatten_parameters()
        rnn_out, states = self.rnn(att_out)
        # No need for unpacking biRNN for pre-attention
        # if self.unpack_dir:
        #     rnn_out = torch.cat((rnn_out[:, :, :self.n_hiddens], rnn_out[:, :, self.n_hiddens:]), dim=1)

        flatten = self.flatten(rnn_out)
        out = self.dense(flatten)
        return out, self._attention_weights


class TransformerEncoderMLP(nn.Module):
    def __init__(self, embed, embed_size, input_size, num_layers, heads, device, forward_expansion,
                 dropout, encoder_type, mlp_depth, mlp_widths, output_size, seq_len=41, **kwargs):
        super(TransformerEncoderMLP, self).__init__()
        self.input_size = input_size
        self.n_taps = seq_len//2
        if encoder_type == "conv":
            self.transformer_encoder = TransformerConvEncoder(embed, embed_size, input_size, num_layers, heads,
                                                              device=device, **kwargs)
        elif encoder_type == "linear":
            self.transformer_encoder = TransformerLinEncoder(embed=embed, embed_size=embed_size, input_size=input_size,
                                                             num_layers=num_layers, heads=heads, device=device,
                                                             forward_expansion=forward_expansion, dropout=dropout)
        else:
            raise ValueError("Transformer encoder type not supported! Must be linear or cnn.")

        mlp_layers = []
        for i in np.arange(mlp_depth):
            if i == 0:
                mlp_layers.append(nn.Linear(input_size * seq_len, mlp_widths[i]))
            else:
                mlp_layers.append(nn.Linear(mlp_widths[i - 1], mlp_widths[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(mlp_widths[-1], output_size))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        enc = self.transformer_encoder(x, mask=None)
        # flatten = self.flatten(enc)
        enc = enc.transpose(1, 2).reshape(enc.size(0), -1)
        mlp = self.mlp(enc)
        return mlp


class MLP(nn.Module):
    def __init__(self, device, depth, width, input_size, seq_len, output_size):
        super(MLP, self).__init__()
        self.device = device
        self.n_taps = seq_len//2
        self.depth, self.width, self.input_size, self.output_size = depth, width, input_size, output_size
        mlp_layers = []
        for i in np.arange(depth):
            if i == 0:
                mlp_layers.append(nn.Linear(input_size * seq_len, width[i]))
            else:
                mlp_layers.append(nn.Linear(width[i-1], width[i]))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(width[-1], output_size))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # [B, seq_len, n_feature] --> [B, seq_len * n_feature] which squeeze the features of the same element
        # of the sequence together (repeating features), instead of repeating sequence
        x = x.transpose(1,2).reshape(x.size(0), -1)
        out = self.mlp(x)
        return out


class SelfAttMLP(MLP):
    def __init__(self, embed, embed_size, **kwargs):
        super(SelfAttMLP, self).__init__(**kwargs)
        self.attention = SelfAttentionLayer(self.input_size, embed, embed_size)
        if embed:
            self.norm = nn.LayerNorm(embed_size)
        else:
            self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x):
        x_att, attention = self.attention(x, x, x)
        # Add + normalisation
        # x_norm = self.dropout(self.norm(x_att + x))
        # x_norm = self.norm(x_att + x) # failed massively
        # To be added, feed forward and another add+norm
        x_att_flatten = x_att.transpose(1, 2).reshape(x_att.size(0), -1)
        out = self.mlp(x_att_flatten)
        return out, attention



class PostFlatAttRecurrent(SimpAttLayerRecurrent):
    """Post flatten attention layer function as weighted fully connected layer"""
    def __init__(self, post_flat=True, **kwargs):
        super(PostFlatAttRecurrent, self).__init__(post_flat=post_flat, **kwargs)

    def forward(self, x, *args):
        self.rnn.flatten_parameters()
        rnn_out, states = self.rnn(x)
        flatten = self.flatten(rnn_out)
        flatten = flatten.unsqueeze(-1)  # add dim to the last axis to perform matmul in attention layer
        att_out, self._attention_weights = self.att(flatten)
        out = self.dense(att_out)
        return out, self._attention_weights


class SimpleScaledDotProductAttentionRecurrent(AttentionRecurrent):
    # from model.attentions
    def __init__(self, d_model, n_heads=1, **kwargs):
        super(SimpleScaledDotProductAttentionRecurrent, self).__init__(**kwargs)
        self.att = SimplifiedScaledDotProductAttention(d_model, n_heads)

    def forward(self, X, *args):
        self.rnn.flatten_parameters()
        rnn_out, states = self.rnn(X)
        # att_out = self.att(rnn_out)
        att_out, self._attention_weights = self.att(rnn_out, rnn_out, rnn_out)
        flatten = self.flatten(att_out)
        out = self.dense(flatten)
        return out


class SelfAttentionRecurrent(AttentionRecurrent):
    # from model.attentions
    def __init__(self, d_model, d_k, d_v, n_heads=1):
        super(SelfAttentionRecurrent, self).__init__()
        self.att = ScaledDotProductAttention(d_model, d_k, d_v, n_heads)

    def forward(self, X, *args):
        rnn_out, states = self.rnn(X)
        # att_out = self.att(rnn_out)
        att_out, self._attention_weights = self.att(rnn_out, rnn_out, rnn_out)
        flatten = self.flatten(att_out)
        out = self.dense(flatten)
        return out


# Encoder Decoder
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        # init dec_X with zeros dec_X=torch.zeros(BATCH_SIZE, N_ADJACENT_SYMBOLS, N_FEATURES).to("cuda:0")
        # for model_summary (torchinfo), or simply put dec_X in *args, since BATCH_SIZE etc. need to be imported
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""

    def __init__(self, input_size, num_hiddens, num_layers=1, bidirectional=False, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size, num_hiddens, num_layers, batch_first=True,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, X, *args):
        # print(type(self.rnn))
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state


class Seq2SeqDecoder(Decoder):
    """The RNN decoder for sequence to sequence learning."""

    def __init__(self, input_size, num_hiddens, output_size, num_layers=1, bidirectional=False, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(num_hiddens, output_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = X.permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2).permute(1, 0, 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output.permute(1, 0, 2)).permute(1, 0, 2)  # first permute is because of batch_first in rnn
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""

    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    # Follow the template of seq2seqDecoder, but do not increase decoder size
    # i.e. keep decoder the same dim as encoder
    def __init__(self, input_size, num_hiddens, output_size, num_layers=1, bidirectional=False, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.att = SimplifiedScaledDotProductAttention(d_model=input_size + num_hiddens, h=1)
        self.rnn = nn.GRU(input_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(num_hiddens, output_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = X.permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2).permute(1, 0, 2)

        # Before decoder RNN layer
        att_X_and_context = self.att(X_and_context, X_and_context, X_and_context)
        output, state = self.rnn(att_X_and_context, state)
        output = self.dense(output.permute(1, 0, 2)).permute(1, 0, 2)  # first permute is because of batch_first in rnn
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

    @property
    def attention_weights(self):
        return self._attention_weights


class BaselineAttentionBiLSTM(nn.Module):  # check FullyConnectedClassifier
    def __init__(self, device, num_hidden_units=16, input_size=4, output_size=2, num_symbols=41, bidirectional=True):
        super(BaselineAttentionBiLSTM, self).__init__()
        self.device = device
        self.hidden_size = num_hidden_units
        self.num_symbols = num_symbols
        self.lstm_encoder = nn.LSTM(input_size, self.hidden_size, bidirectional=bidirectional,
                                    batch_first=True)  # [input_size, hidden_size]
        decoder_input = 2 * self.hidden_size if bidirectional else self.hidden_size
        self.lstm_decoder = nn.LSTM(decoder_input, self.hidden_size, bidirectional=bidirectional,
                                    batch_first=True)  # [hidden_size, hidden_size]
        self.attention = SimpleScaledDotProductAttentionRecurrent(self.hidden_size, self.num_symbols)
        self.flatten = Flatten()
        self.linear = nn.Linear(self.num_symbols * 2 * self.hidden_size, output_size) if bidirectional \
            else nn.Linear(self.num_symbols * self.hidden_size, output_size)
        # in_feature (flattened combining hidden units and seq_length), out_feature, bidirectional

    def forward(self, x):
        out, _ = self.lstm_encoder(x)  # LSTM output: tuple(output,(h_n, c_n)), using "_" to ignore the second output
        out = self.attention(out)
        out, _ = self.lstm_decoder(out)
        out = self.flatten(out)  # reshape for flattening according to batch_size => out = out.reshape(x.shape[0], -1)
        out = self.linear(out)
        return out


class BaselineAttentionBiGRU(nn.Module):  # check FullyConnectedClassifier
    def __init__(self, device, num_hidden_units=16, input_size=4, output_size=2, num_symbols=41, bidirectional=True):
        super(BaselineAttentionBiGRU, self).__init__()
        self.device = device
        self.hidden_size = num_hidden_units
        self.num_symbols = num_symbols
        self.gru_encoder = nn.GRU(input_size, self.hidden_size, bidirectional=bidirectional,
                                  batch_first=True)  # [input_size, hidden_size]
        decoder_input = 2 * self.hidden_size if bidirectional else self.hidden_size
        self.gru_decoder = nn.GRU(decoder_input, self.hidden_size, bidirectional=bidirectional,
                                  batch_first=True)  # [hidden_size, hidden_size]
        self.attention = SimpleScaledDotProductAttentionRecurrent(self.hidden_size, self.num_symbols)
        self.flatten = Flatten()
        self.linear = nn.Linear(self.num_symbols * 2 * self.hidden_size, output_size) if bidirectional \
            else nn.Linear(self.num_symbols * self.hidden_size, output_size)
        # in_feature (flattened combining hidden units and seq_length), out_feature, bidirectional

    def forward(self, x):
        out, _ = self.gru_encoder(x)  # LSTM output: tuple(output,(h_n, c_n)), using "_" to ignore the second output
        out = self.attention(out)
        out, _ = self.gru_decoder(out)
        out = self.flatten(out)  # reshape for flattening according to batch_size => out = out.reshape(x.shape[0], -1)
        out = self.linear(out)
        return out
