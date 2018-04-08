import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init
from LSTM_KE import LSTMKECell


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
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1) * mask
        alpha = F.softmax(attn_energies, dim=-1)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        print (alpha[0])
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

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUDualAttnEnc, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
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
        c = torch.cat([c[0], c[1]], dim=-1)
        r = torch.cat([r[0], r[1]], dim=-1)

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
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
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


class GRUAttn_KeyCNN(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttn_KeyCNN, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=2*emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.key_rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('ubuntu_data/man_d.npy').item()
        self.tech_w = 0.0
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
        key_c, key_r = self.get_weighted_key(x1, x2)
        sc, c, r = self.forward_enc(x1, x2, key_c, key_r)
        c_attn = self.forward_attn(sc, r, x1mask)

        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_key(self, context):

        key_mask = torch.zeros(context.size(0), 1)
        keys = torch.zeros(context.size(0), 44)
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_mask[i][j] = 1
                    keys[i] = torch.from_numpy(self.ubuntu_cmd_vec[word]).type(torch.LongTensor)
        return Variable(key_mask.cuda()), Variable(keys.type(torch.LongTensor).cuda())

    def get_weighted_key(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        key_mask_c, keys_c = self.forward_key(x1)
        key_mask_r, keys_r = self.forward_key(x2)
        print (keys_c)
        key_emb_c = self.word_embed(keys_c)
        key_emb_r = self.word_embed(keys_r)
        _, key_emb_c = self.key_rnn(key_emb_c)
        _, key_emb_r = self.key_rnn(key_emb_r)
        key_emb_c = key_emb_c * key_mask_c
        key_emb_r = key_emb_r * key_mask_r

        return key_emb_c, key_emb_r

    def forward_enc(self, x1, x2, key_emb_c, key_emb_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        x2_emb = self.emb_drop(self.word_embed(x2))
        x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) #concat the bi directional stuffs
        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x2).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
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

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, key_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttnmitKey, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        self.key_emb = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        if key_emb is not None:
            self.key_emb.weight.data.copy_(key_emb)

        self.key_emb.weight.requires_grad = False

        self.rnn = nn.GRU(
            input_size=emb_dim + 50, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True, dropout=0.2
        )

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.key_wghtc = nn.Linear(200, 50)
        self.key_wghtr = nn.Linear(200, 50)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('ubuntu_data/man_dict_vec.npy').item()
        self.tech_w = 0.0
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
        key_c, key_r = self.get_weighted_key(x1, x2)
        sc, c, r = self.forward_enc(x1, x2, key_c, key_r)
        c_attn = self.forward_attn(sc, r, x1mask)

        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_key(self, context):
        key_emb = torch.zeros(context.size(0), context.size(1))
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_emb[i][j] = 1
        return Variable(key_emb.cuda())

    def check_key(self, context):
        key_emb = torch.zeros(context.size(0))
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_emb[i] = 1

        return Variable(key_emb.cuda())

    def get_weighted_key(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        key_mask_c = self.forward_key(x1)
        key_mask_r = self.forward_key(x2)
        key_emb_c = (self.key_emb(x1))
        key_emb_r = (self.key_emb(x2))
        key_emb_c = self.key_wghtc(key_emb_c)
        key_emb_r = self.key_wghtr(key_emb_r)
        key_emb_c = key_emb_c * key_mask_c.unsqueeze(2).repeat(1, 1, 50)
        key_emb_r = key_emb_r * key_mask_r.unsqueeze(2).repeat(1, 1, 50)

        return key_emb_c, key_emb_r

    def forward_enc(self, x1, x2, key_emb_c, key_emb_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        x2_emb = self.emb_drop(self.word_embed(x2))
        x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) #concat the bi directional stuffs
        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B*T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x2).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
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


class LSTMKeyAttn(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(LSTMKeyAttn, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = LSTMKECell(
            input_size=emb_dim, hidden_size=h_dim,
            topic_size=200
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
        self.ubuntu_cmd_vec = np.load('ubuntu_data/man_dict_vec.npy').item()
        self.ubuntu_cmds = np.load('ubuntu_data/man_dict.npy').item()
        self.h_dim = h_dim
        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        #size = self.rnn.bias_hh_l0.size(0)
        #self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        #size = self.rnn.bias_ih_l0.size(0)
        #self.rnn.bias_ih_l0.data[size//4:size//2] = 2

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

    def forward_key(self, context):
        key_emb = torch.zeros(context.size(0), context.size(1), 200)
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                #torch_val = torch.zeros(200)
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
        key_emb_c = self.forward_key(x1)
        key_emb_r = self.forward_key(x2)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        hxc = Variable(torch.zeros(x1.size(0), self.h_dim)).cuda()
        cxc = Variable(torch.zeros(x1.size(0), self.h_dim)).cuda()
        hxr = Variable(torch.zeros(x1.size(0), self.h_dim)).cuda()
        cxr = Variable(torch.zeros(x1.size(0), self.h_dim)).cuda()
        sc = []
        r = []
        for j in range(x1_emb.size(1)):
            hxc, cxc = self.rnn(torch.squeeze(x1_emb[:, j: j+1], 1), torch.squeeze(key_emb_c[:, j: j+1], 1), (hxc, cxc))
            sc.append(hxc)

        for j in range(x2_emb.size(1)):
            hxr, cxr = self.rnn(torch.squeeze(x2_emb[:, j: j+1], 1), torch.squeeze(key_emb_r[:, j: j+1], 1), (hxr, cxr))
            r.append(hxr)

        return torch.stack(sc).transpose(0, 1), sc[-1], r[-1]  # B X T X H, B X H, B X H

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
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = attn.bmm(x).transpose(1, 2) #B,T,D * B,D,1 --> B,1,T
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
