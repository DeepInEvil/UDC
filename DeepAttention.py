import numpy as np
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

    def forward(self, x1, x2, x1mask, x2mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, sr, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask)

        o = self.forward_fc(c_attn, r_attn)
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
        sr, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1)
        r = torch.cat([r[0], r[1]], dim=-1)

        #return sc, c.squeeze(), r.squeeze()
        return sc, sr, c.squeeze(), r.squeeze()

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
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
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
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=50,
            num_layers=1, batch_first=True, bidirectional=True
        )

        # self.rnn_key = nn.GRU(
        #     input_size=emb_dim, hidden_size=50,
        #     num_layers=1, batch_first=True, bidirectional=True
        # )

        #self.n_filter = 30
        self.h_dim = h_dim

        # self.conv3 = nn.Conv2d(1, self.n_filter, (3, emb_dim))
        # self.conv4 = nn.Conv2d(1, self.n_filter, (4, emb_dim))
        # self.conv5 = nn.Conv2d(1, self.n_filter, (5, emb_dim))

        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim + 100))
        #self.fc_key = nn.Parameter(torch.FloatTensor(2*h_dim + 100, 2*h_dim))
        #self.M = nn.Parameter(torch.FloatTensor(2*h_dim + 50*2, 2*h_dim + 50*2))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('ubuntu_data/command_desc_dict.npy').item()
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

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

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
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)

        o = self.forward_fc(c_attn, r, key_c, key_r)

        return o.view(-1)

    def forward_key(self, context):

        key_mask = torch.zeros(context.size(0), 100)
        keys = torch.zeros(context.size(0), 320)
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_mask[i] = 1
                    keys[i] = torch.from_numpy(self.ubuntu_cmd_vec[word]).type(torch.LongTensor)
        return Variable(key_mask.cuda()), Variable(keys.type(torch.LongTensor).cuda())

    def get_weighted_key(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        key_mask_c, keys_c = self.forward_key(x1)
        key_mask_r, keys_r = self.forward_key(x2)
        key_emb_c = self.word_embed(keys_c)
        key_emb_r = self.word_embed(keys_r)
        key_emb_c = self._forward(key_emb_c)
        key_emb_r = self._forward(key_emb_r)
        #key_emb_c = key_emb_c.squeeze().unsqueeze(1).repeat(1, x1.size(1), 1) * key_mask_c.unsqueeze(2).repeat(1, 1, 100)
        #key_emb_r = key_emb_r.squeeze().unsqueeze(1).repeat(1, x2.size(1), 1) * key_mask_r.unsqueeze(2).repeat(1, 1, 100)
        return key_emb_c, key_emb_r
        #return key_emb_r

    def _forward(self, x):
        # x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
        #
        # x3 = F.relu(self.conv3(x)).squeeze()
        # x4 = F.relu(self.conv4(x)).squeeze()
        # x5 = F.relu(self.conv5(x)).squeeze()
        #
        # # Max-over-time-pool
        # x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        # x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        # x5 = F.max_pool1d(x5, x5.size(2)).squeeze()
        #
        # out = torch.cat([x3, x4, x5], dim=1)
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        #x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        x2_emb = self.emb_drop(self.word_embed(x2))
        #x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) # concat the bi-directional hidden layers
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
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r, key_c, key_r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        # (batch_size x 1 x h_dim)
        #c = torch.cat([c, key_c], dim=-1)
        s = torch.cat([c, key_r], dim=-1)
        s = F.tanh(s)
        s = s * torch.cat([c, key_r], dim=-1) + (1 - s) * torch.cat([r, key_c], dim=-1)
        #r = torch.cat([r, s], dim=-1)
        o = torch.mm(c, self.M).unsqueeze(1)
        #o_fc = torch.mm(s, self.fc_key)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, s.unsqueeze(2))
        o = o + self.b

        return o


class GRUAttn_KeyCNN2(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttn_KeyCNN2, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        #self.desc_emb = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx, sparse=True)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        self.desc_rnn_size = 100
        self.n_filter = 50
        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )
        # self.rnn_key = nn.GRU(
        #     input_size=emb_dim, hidden_size=50,
        #     num_layers=1, batch_first=True, bidirectional=True
        # )

        self.h_dim = h_dim
        #
        # self.conv1 = nn.Conv2d(1, self.n_filter, (1, emb_dim))
        # self.conv3 = nn.Conv2d(1, self.n_filter, (3, emb_dim))
        # self.conv5 = nn.Conv2d(1, self.n_filter, (5, emb_dim))
        # self.conv7 = nn.Conv2d(1, self.n_filter, (7, emb_dim))

        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        #self.fc_key = nn.Parameter(torch.FloatTensor(2*h_dim + 100, 2*h_dim))
        #self.M = nn.Parameter(torch.FloatTensor(2*h_dim + 50*2, 2*h_dim + 50*2))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('ubuntu_data/command_desc_dict.npy').item()
        self.tech_w = 0.0
        if gpu:
            self.cuda()

    def custom_replace(self, tensor, on_zero=1, on_non_zero=0):
        # we create a copy of the original tensor,
        # because of the way we are replacing them.
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask, x2mask, key_r, key_mask_r):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        #print (torch.sum(key_mask_r != 0))
        #key_mask_c = key_mask_c.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        key_mask_r = key_mask_r.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        key_emb_r = self.get_weighted_key(key_r, key_mask_r)
        sc, sr, c, r = self.forward_enc(x1, x2, key_emb_r)
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask)
        o = self.forward_fc(c_attn, r_attn)

        return o.view(-1)

    def get_weighted_key(self, key_r, key_mask_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        #mask_c, keys_c = self.forward_key(x1, 80)
        #mask_c = mask_c.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        # key_emb_c = Variable(torch.zeros(key_c.size(0), key_c.size(1), self.n_filter * 4)).cuda()
        # #key_emb_c = Variable(torch.Tensor([[self._forward(self.word_embed(key_c[b])) for b in range(key_c.size(0))]]))
        # for b in range(key_c.size(0)):
        #     emb = self.emb_drop(self.word_embed(key_c[b]))
        #     key_emb_c[b] = self._forward(emb)
        # b_s = key_c.size(0)
        # s_len = key_c.size(1)
        # key_emb = self.emb_drop(self.word_embed(key_c.view(b_s*s_len, -1)))
        # key_emb_c = self._forward(key_emb).view(b_s, s_len, -1)
        # key_emb_c = key_emb_c * key_mask_c
        # del(key_emb, b_s, s_len)
        #mask_r, keys_r = self.forward_key(x2, 80)
        #mask_r = mask_r.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        # key_emb_r = Variable(torch.zeros(key_r.size(0), key_r.size(1), self.n_filter * 4)).cuda()
        # for b in range(key_r.size(0)):
        #     emb = self.emb_drop(self.word_embed(key_r[b]))
        #     key_emb_r[b] = self._forward(emb)
        b_s = key_r.size(0)
        s_len = key_r.size(1)
        key_emb = self.emb_drop(self.word_embed(key_r.view(b_s * s_len, -1)))
        #_, h = self.rnn_desc(key_emb)
        #key_emb = torch.cat([h[0], h[1]], dim=-1).view(b_s, s_len, -1)
        #print (key_emb.size(), key_mask_r.size())
        #key_emb_r = key_emb * key_mask_r
        key_emb = self._forward(key_emb)
        key_emb_r = key_emb.view(b_s, s_len, -1) * key_mask_r
        del (key_emb, b_s, s_len)

        #return key_emb_c, key_emb_r
        return key_emb_r

    def _forward(self, x):
        # x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
        #
        # x1 = F.relu(self.conv1(x)).squeeze()
        # x3 = F.relu(self.conv3(x)).squeeze()
        # x5 = F.relu(self.conv5(x)).squeeze()
        # x7 = F.relu(self.conv7(x)).squeeze()
        #
        # # Max-over-time-pool
        # x1 = F.max_pool1d(x1, x1.size(2)).squeeze()
        # x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        # x5 = F.max_pool1d(x5, x5.size(2)).squeeze()
        # x7 = F.max_pool1d(x7, x7.size(2)).squeeze()
        #
        # out = torch.cat([x1, x3, x5, x7], dim=1)
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out.squeeze()

    def forward_enc(self, x1, x2, key_emb_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        #x1_emb = x1_emb + key_emb_r
        #x1_emb = x1_emb * (1 - maskc) + key_emb_c
        #x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        x2_emb = self.emb_drop(self.word_embed(x2))
        x2_emb = x2_emb + key_emb_r
        #x2_emb = x2_emb * (1 - maskr) + key_emb_r
        #x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        sr, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) # concat the bi-directional hidden layers
        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, sr, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B, 1, T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """

        #_, key_c = self.rnn_desc(key_emb_c)
        #_, key_r = self.rnn_desc(key_emb_r)
        # (batch_size x 1 x h_dim)
        #key_c = key_c.squeeze()
        #key_r = key_r.squeeze()
        #c = torch.cat([c, key_emb_r], dim=-1)
        #s = torch.cat([c, key_emb_r], dim=-1)
        #s = F.tanh(s)
        #s = s * torch.cat([c, key_emb_r], dim=-1) + (1 - s) * torch.cat([r, key_emb_r], dim=-1)
        #r = torch.cat([r, s], dim=-1)
        #print (s.size())
        o = torch.mm(c, self.M).unsqueeze(1)
        #print (o.size(  ))
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o


class GRUAttn_KeyCNN4(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttn_KeyCNN4, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        self.desc_rnn_size = 100
        self.n_filter = 50
        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.Wc = nn.Parameter(torch.FloatTensor(2*h_dim, emb_dim))
        self.We = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.init_params_()
        self.tech_w = 0.0
        if gpu:
            self.cuda()

    def custom_replace(self, tensor, on_zero=1, on_non_zero=0):
        # we create a copy of the original tensor,
        # because of the way we are replacing them.
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask, x2mask, key_r, key_mask_r):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        key_mask_r = key_mask_r.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        #git pprint (torch.sum(key_mask_r))
        key_emb_r = self.get_weighted_key(key_r, key_mask_r)
        sc, sr, c, r = self.forward_enc(x1, x2, key_emb_r)
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask)
        o = self.forward_fc(c_attn, r_attn)

        return o.view(-1)

    def get_weighted_key(self, key_r, key_mask_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        b_s = key_r.size(0)
        s_len = key_r.size(1)
        key_emb = self.emb_drop(self.word_embed(key_r.view(b_s * s_len, -1)))
        key_emb = self._forward(key_emb)
        key_emb_r = key_emb.view(b_s, s_len, -1) * key_mask_r
        del (key_emb, b_s, s_len)

        return key_emb_r

    def _forward(self, x):
        """
        get description embeddings
        :param x:
        :return:
        """
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out.squeeze()

    def forward_enc(self, x1, x2, key_emb_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        b, s = x2.size(0), x2.size(1)
        x1_emb = self.emb_drop(self.word_embed(x1)) # B X S X E
        sc, c = self.rnn(x1_emb)
        c = torch.cat([c[0], c[1]], dim=-1)  # concat the bi-directional hidden layers, shape = B X H

        c_k = c.unsqueeze(1).repeat(1, key_emb_r.size(1), 1)

        x2_emb = self.emb_drop(self.word_embed(x2))
        alpha_dk = F.softmax(torch.mm(c_k.view(b*s, -1), self.Wc).view(b, s, self.emb_dim) + torch.mm(key_emb_r.view(b*s, -1), self.We).view(b, s, self.emb_dim), dim=-1)

        x2_emb = (1 - alpha_dk) * x2_emb + alpha_dk * key_emb_r
        # Each is (1 x batch_size x h_dim)

        sr, r = self.rnn(x2_emb)

        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, sr, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o

class GRUAttn_KeyCNN3(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttn_KeyCNN3, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        #self.desc_emb = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx, sparse=True)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        self.desc_rnn_size = 100
        self.n_filter = 50
        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )
        # self.rnn_key = nn.GRU(
        #     input_size=emb_dim, hidden_size=50,
        #     num_layers=1, batch_first=True, bidirectional=True
        # )

        self.h_dim = h_dim

        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.desc_rnn_add = nn.Linear(2*self.desc_rnn_size, 2*self.h_dim)
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('ubuntu_data/command_desc_dict.npy').item()
        self.tech_w = 0.0
        if gpu:
            self.cuda()

    def custom_replace(self, tensor, on_zero=1, on_non_zero=0):
        # we create a copy of the original tensor,
        # because of the way we are replacing them.
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask, x2mask, key_r, key_mask_r):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        #print (torch.sum(key_mask_r != 0))
        key_mask_r = key_mask_r.unsqueeze(2).repeat(1, 1, self.n_filter * 4)
        key_emb_r = self.get_weighted_key(key_r, key_mask_r)
        sc, sr, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask, key_emb_r)
        o = self.forward_fc(c_attn, r_attn)

        return o.view(-1)

    def get_desc(self, word, max_len):
        try:
            return self.ubuntu_cmd_vec[word][:max_len]
        except KeyError:
            return [0] * max_len

    def get_weighted_key(self, key_r, key_mask_r):
        """
        key_r, key_mask_r: seqs of words (batch_size, seq_len), (B X S X E)
        """

        b_s = key_r.size(0)
        s_len = key_r.size(1)
        key_emb = self.emb_drop(self.word_embed(key_r.view(b_s * s_len, -1)))  # BXS, KS, E

        key_emb = self._forward(key_emb)
        key_emb_r = key_emb.view(b_s, s_len, -1) * key_mask_r  # B, S, E
        del (key_emb, b_s, s_len)

        return key_emb_r

    def _forward(self, x):
        """

        :param x: B X S X E
        :return:
        """
        #_, h = self.rnn_desc(x)
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out.squeeze()

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        sr, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) # concat the bi-directional hidden layers
        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, sr, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask, key_emb=None):
        """
        get attention vectors
        :param x1:
        :param x2:
        :param mask:
        :param key_emb:
        :return:
        """
        if key_emb is not None:
            key_emb = F.tanh(self.desc_rnn_add(key_emb.view(key_emb.size(0) * key_emb.size(1), -1))).view(key_emb.size(0),
                                                                                                  key_emb.size(1), -1)
            x1 = x1 + key_emb
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))  # B, T, D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2))  # B, T, D * B, D, 1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
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

class GRUAttn_KeyCNN_AllKeys(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(GRUAttn_KeyCNN_AllKeys, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.h_dim = h_dim

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.n_filter = 30
        self.h_key_dim = self.n_filter * 3

        #self.key_rnn = nn.GRU(input_size=emb_dim, hidden_size=50, batch_first=True)

        self.emb_drop = nn.Dropout(emb_drop)
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.scale = 1. / math.sqrt(max_seq_len)
        self.out_hidden = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.init_params_()
        self.ubuntu_cmd_vec = np.load('/home/DebanjanChaudhuri/UDC/ubuntu_data/man_d.npy').item()
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
        key_r = self.get_weighted_key(x1, x2)
        sc, c, r = self.forward_enc(x1, x2, key_r)
        c_attn = self.forward_attn(sc, r, x1mask)

        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_key(self, context):

        key_mask = torch.zeros(context.size(0), context.size(1))
        keys = torch.zeros(context.size(0), context.size(1), 44)  # batch x seq x desc_len
        for i in range(context.size(0)):
            utrncs = context[i].cpu().data.numpy()
            for j, word in enumerate(utrncs):
                if word in self.ubuntu_cmd_vec.keys():
                    key_mask[i][j] = 1
                    keys[i, j] = torch.from_numpy(self.ubuntu_cmd_vec[word]).type(torch.LongTensor)
        return Variable(key_mask.cuda()), Variable(keys.type(torch.LongTensor).cuda())

    def get_weighted_key(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        key_mask_c, keys_c = self.forward_key(x1)
        key_mask_r, keys_r = self.forward_key(x2)

        # print(keys_c.size(), keys_r.size())

        keys_emb_c = Variable(torch.zeros(keys_c.size(0), keys_c.size(1), self.h_dim)).cuda()

        for i, key_c in enumerate(keys_c.transpose(0, 1)):  # iterate sequence
            # key_c: (batch, desc_len)
            key_emb_c = self.word_embed(key_c)
            # keys_emb_c[:, i, :] = self._forward(key_emb_c)
            # _, key_emb_c =
            _, k_c = self.rnn(key_emb_c)
            k_c = torch.cat([k_c[0], k_c[1]], dim=-1)
            keys_emb_c[:, i, :] = k_c

        keys_emb_r = Variable(torch.zeros(keys_r.size(0), keys_r.size(1), self.h_dim)).cuda()

        for i, key_r in enumerate(keys_r.transpose(0, 1)):  # iterate sequence
            # key_r: (batch, desc_len)
            key_emb_r = self.word_embed(key_r)
            # keys_emb_r[:, i, :] = self._forward(key_emb_r)
            _, k_c = self.rnn(key_emb_r)
            k_c = torch.cat([k_c[0], k_c[1]], dim=-1)
            keys_emb_c[:, i, :] = k_c
            keys_emb_r[:, i, :] = self.rnn(key_emb_r)[1].squeeze()

        # print(key_emb_c.size(), key_emb_r.size())

        # key_emb_c = key_emb_c.squeeze().unsqueeze(1).repeat(1, x1.size(1), 1) * key_mask_c.unsqueeze(2).repeat(1, 1, self.n_filter * 3)
        # key_emb_r = key_emb_r.squeeze().unsqueeze(1).repeat(1, x2.size(1), 1) * key_mask_r.unsqueeze(2).repeat(1, 1, self.n_filter * 3)

        # print(keys_emb_c.size(), keys_emb_r.size())

        return keys_emb_c, keys_emb_r

    def forward_enc(self, x1, x2, key_emb_r):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x1_emb = torch.cat([x1_emb, key_emb_c], dim=-1)
        x2_emb = self.emb_drop(self.word_embed(x2))
        x2_emb = torch.cat([x2_emb, key_emb_r], dim=-1)

        # print(x1_emb.size(), x2_emb.size())

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        c = torch.cat([c[0], c[1]], dim=-1) #concat the bi directional stuffs
        r = torch.cat([r[0], r[1]], dim=-1)

        # print(c.size(), r.size())

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
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
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
        print (alpha)
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
