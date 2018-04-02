import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init


class LSTMDualAttnEnc(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.3, max_seq_len=160):
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
        self.attn = nn.Linear(h_dim, h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        #self.attn_out = nn.Linear(h_dim, 1)
        self.softmax = nn.Softmax()
        self.init_params_()
        #self.bn = nn.BatchN

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        o = self.forward_fc(c_attn, r)
        #print (c_attn.size())

        #o = F.tanh(self.out_hidden(c_attn))
        #o = self.out_drop(o)
        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        sc, (c, _) = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x = x.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        #print (attn.size(), x.size())
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
        #print (attn_energies.size())
        attn_energies = attn_energies.squeeze(1) * mask
        #attn_energies = attn_energies.squeeze(1).masked_fill(mask, -1e12)
        alpha = F.softmax(attn_energies, dim=-1)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        print (alpha[0])
        #print (alpha.size(), x1.size())
        weighted_attn = alpha.bmm(x1)


        return weighted_attn.squeeze()


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


class LSTMPAttn(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.5):
        super(LSTMPAttn, self).__init__()

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
        self.attn = nn.Linear(h_dim, h_dim)
        #self.attn_out = nn.Linear(h_dim, 1)
        self.softmax = nn.Softmax()
        self.init_params_()
        #self.max_seq_len = max_seq_len
        self.out_h = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.2)
        self.nl = nn.Tanh()
        self.h_dim = h_dim

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
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
        sc, (c, _) = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        return sc, c, r.squeeze()

    # def forward_attn(self, x1, x, mask):
    #     """
    #     attention
    #     :param x1: batch X seq_len X dim
    #     :return:
    #     """
    #     max_len = x1.size(1)
    #     b_size = x1.size(0)
    #
    #     x = x.squeeze(0).unsqueeze(2)
    #     attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
    #     #print (attn.size(), x.size())
    #     attn = attn.view(b_size, max_len, -1) # B,T,D
    #     attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
    #     #print (attn_energies.size())
    #     attn_energies = attn_energies.squeeze(1).masked_fill(mask, -1e12)
    #     alpha = F.softmax(attn_energies, dim=-1)  # B,T
    #     alpha = alpha.unsqueeze(1)  # B,1,T
    #     #print (alpha.size(), x1.size())
    #     weighted_attn = alpha.bmm(x1)
    #
    #     return weighted_attn.squeeze()

    def forward_attn(self, x1, x, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x = x.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,T
        attn = attn.view(b_size, -1)
        print (attn.size(), x.size())
        attn = attn.squeeze(1).masked_fill(mask, -1e12)
        alpha = F.softmax(attn, dim=-1)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        #print (alpha.size(), x1.size())
        weighted_attn = alpha.bmm(x1)

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        results = []

        # (batch_size x 1 x h_dim)
        for i in range(len(c)):

            context_h = c[i].view(1, self.h_dim)
            response_h = r[i]
            w_mm = torch.mm(context_h, self.M).squeeze()
            ans = self.nl(w_mm * response_h)
            results.append((self.out_h(ans)))

        o = torch.stack(results)
        return o.squeeze()


class GRUDualAttnEnc(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.7, max_seq_len=160):
        super(GRUDualAttnEnc, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(h_dim, h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        #self.attn_out = nn.Linear(h_dim, 1)
        self.softmax = nn.Softmax()
        self.init_params_()
        #self.bn = nn.BatchN

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        #init.xavier_uniform(self.attn.data)

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        o = self.forward_fc(c_attn, r)
        #print (c_attn.size())

        #o = F.tanh(self.out_hidden(c_attn))
        #o = self.out_drop(o)
        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x = x.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        #print (attn.size(), x.size())
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
        #print (attn_energies.size())
        #attn_energies = attn_energies.squeeze(1).masked_fill(mask, -float('inf'))
        #attn_energies = attn_energies * mask
        #print (attn_energies.size(), mask.size())
        attn_energies = attn_energies.squeeze(1) * mask  # B, T
        alpha = F.softmax(attn_energies, dim=-1)  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

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


class GRUAttnmitKey(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.7, max_seq_len=160):
        super(GRUAttnmitKey, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(h_dim+200, h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        #self.attn_out = nn.Linear(h_dim, 1)
        self.softmax = nn.Softmax()
        self.init_params_()
        #self.bn = nn.BatchN
        self.ubuntu_cmd_vec = np.load('ubuntu_data/man_dict_vec.npy').item()
        self.ubuntu_cmds = np.load('ubuntu_data/man_dict.npy').item()

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        #init.xavier_uniform(self.attn.data)

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        key_emb_c = self.forward_key(x1)
        key_emb_r = self.forward_key(x2)
        sc = torch.cat([sc, key_emb_c], dim=-1)
        print (sc.size())
        c_attn = self.forward_attn(sc, r, x1mask)
        print (c_attn.size())
        o = self.forward_fc(c_attn, r)
        #print (c_attn.size())

        #o = F.tanh(self.out_hidden(c_attn))
        #o = self.out_drop(o)
        return o.view(-1)

    def forward_key(self, context):
        key_emb = torch.zeros(context.size(0), context.size(1), 200)
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_emb[i][j] = torch.from_numpy(self.ubuntu_cmd_vec[word])

        return Variable(key_emb.cuda())

    def check_key(self, context):
        key_emb = torch.zeros(context.size(0))
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_emb[i] = 1

        return Variable(key_emb.cuda())

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        #x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        #print (x1_emb[0])
        x2_emb = self.emb_drop(self.word_embed(x2))
        #x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x = x.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        #print (attn.size(), x.size())
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
        #print (attn_energies.size())
        #attn_energies = attn_energies.squeeze(1).masked_fill(mask, -float('inf'))
        #attn_energies = attn_energies * mask
        #print (attn_energies.size(), mask.size())
        attn_energies = attn_energies.squeeze(1) * mask  # B, T
        alpha = F.softmax(attn_energies, dim=-1)  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

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