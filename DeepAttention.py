import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# ## Functions to accomplish attention

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if nonlinearity=='tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity=='tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None :
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


class LSTMDualAttnEnc(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.5):
        super(LSTMDualAttnEnc, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(torch.FloatTensor(h_dim, h_dim))
        self.attn_out = nn.Parameter(torch.Tensor(h_dim, 1))
        self.softmax = nn.Softmax()
        self.init_params_()

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2
        self.attn.data.uniform_(-0.1, 0.1)
        self.attn_out.data.uniform_(-0.1, 0.1)

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(c)
        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        c, _ = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        return c, r.squeeze()

    def forward_attn(self, x1):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(self.attn_out).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)

        return weighted_attn

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        # (batch_size x 1 x h_dim)
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o