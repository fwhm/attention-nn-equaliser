import torch
import torch.nn as nn
from torch.nn import init


class Flatten(nn.Module):
    # def __init__(self, **kwargs):
    #     super(Flatten, self).__init__()
    #     if kwargs.get('seq_len') is not None:
    #         self.seq_len = kwargs.get('seq_len')
    #     if kwargs.get('n_hiddens') is not None:
    #         self.n_hiddens = kwargs.get('n_hiddens')
    #     if kwargs.get('bidirectional') is not None:
    #         self.bidirectional = kwargs.get('bidirectional')
    #         self.n_hiddens = 2 * kwargs.get('n_hiddens') if self.bidirectional else kwargs.get('n_hiddens')

    def forward(self, inputs):
        return inputs.reshape(inputs.size(0), -1)
        # return input.view(input.size(0), -1)
        # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans
        # across two contiguous subspaces). Use .reshape(...) instead.


# Trim RNN output
class TrimRnnOutTapsLayer(nn.Module):
    def __init__(self, trim_tap):
        super(TrimRnnOutTapsLayer, self).__init__()
        self.trim_tap = trim_tap

    def forward(self, rnn_outputs):
        mid_idx = rnn_outputs.shape[1]//2
        return rnn_outputs[:, mid_idx-self.trim_tap:mid_idx+self.trim_tap+1, :]


# ## Attention Components ## #
class SimpAttLayer(nn.Module):
    # My own very first simple attention layer
    def __init__(self, hidden_size, seq_len, bidirectional, unpack_dir, return_sequence=True):
        super(SimpAttLayer, self).__init__()
        if bidirectional:
            if unpack_dir:
                seq_len = 2 * seq_len
                self.att_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
            else:
                self.att_weights = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        else:
            self.att_weights = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.att_bias = nn.Parameter(torch.Tensor(seq_len, 1))
        self.return_sequence = return_sequence
        self.init_att_weights()

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.att_weights) + self.att_bias)
        att = torch.softmax(e, dim=1)
        # if self.post_flat: att = att.unsqueeze(-1)  # expand dim for post flatten attention layer
        output = x * att
        # if self.post_flat: output = output.squeeze()  # squeeze dim for post flatten attention layer
        if not self.return_sequence:
            return torch.sum(output, dim=1)  # for non-return sequence
        return output, att

    def init_att_weights(self):
        init.normal_(self.att_weights, std=0.001)
        if self.att_bias is not None:
            init.constant_(self.att_bias, 0)


class SimpAttInputLayer(nn.Module):
    # Simple attention layer before RNN, on the input sequence
    def __init__(self, input_size, seq_len, return_sequence=True):
        super(SimpAttInputLayer, self).__init__()
        self.att_weights = nn.Parameter(torch.Tensor(input_size, 1))

        self.att_bias = nn.Parameter(torch.Tensor(seq_len, 1))
        self.return_sequence = return_sequence
        self.init_att_weights()

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.att_weights) + self.att_bias)
        att = torch.softmax(e, dim=1)
        # if self.post_flat: att = att.unsqueeze(-1)  # expand dim for post flatten attention layer
        output = x * att
        # if self.post_flat: output = output.squeeze()  # squeeze dim for post flatten attention layer
        if not self.return_sequence:
            return torch.sum(output, dim=1)  # for non-return sequence
        return output, att

    def init_att_weights(self):
        init.normal_(self.att_weights, std=0.001)
        if self.att_bias is not None:
            init.constant_(self.att_bias, 0)


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size, embed, embed_size):
        super(SelfAttentionLayer, self).__init__()
        self.input_size = input_size
        if embed:
            self.embed_size = embed_size
        else:
            self.embed_size = input_size
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)


    def forward(self, values, keys, query):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, dim)
        keys = self.keys(keys)  # (N, key_len, dim)
        queries = self.queries(query)  # (N, query_len, dim)

        energy = torch.einsum("nqd,nkd->nqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # (N, query_len, key_len)

        out = torch.einsum("nql,nld->nqd", [attention, values])  # # (N, query_len, dim)

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out, attention


"""
class MLP(nn.Module):
    # to be continued
    def __init__(self, depth, width, activation_func):
        super(MLP, self).__init__()
        mlp_layers = []
        if activation_func == "relu":
            activation = nn.ReLU()
        elif activation_func == "softmax":
            activation = nn.Softmax()

        for i in np.arange(depth):
            mlp_layers.append(nn.Linear(width[i], width[i + 1]))
            mlp_layers.append(activation)

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self):
        return self.mlp
"""
