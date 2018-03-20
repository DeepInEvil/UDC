import torch
import torch.nn as nn
import torch.nn.functional as F
from data import position_encoding_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#from Deep_attention import SelfAttention


class CNNDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=300, pretrained_emb=None, gpu=False):
        super(CNNDualEncoder, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.n_filter = h_dim // 3
        self.h_dim = self.n_filter * 3

        self.conv3 = nn.Conv2d(1, self.n_filter, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, self.n_filter, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, self.n_filter, (5, emb_dim))

        self.M = nn.Parameter(nn.init.xavier_normal(torch.FloatTensor(self.h_dim, self.h_dim)))
        self.b = nn.Parameter(torch.FloatTensor([0]))

        if gpu:
            self.cuda()

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: tensors of (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        emb_x1 = self.word_embed(x1)
        emb_x2 = self.word_embed(x2)

        c1 = self._forward(emb_x1)
        # (batch_size x h_dim x 1)
        c2 = self._forward(emb_x2).unsqueeze(2)

        # (batch_size x 1 x h_dim)
        o = torch.mm(c1, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, c2)
        o = o + self.b

        return o.view(-1)

    def _forward(self, x):
        x = x.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(x)).squeeze()
        x4 = F.relu(self.conv4(x)).squeeze()
        x5 = F.relu(self.conv5(x)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        out = torch.cat([x3, x4, x5], dim=1)

        return out


class LSTMDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False):
        super(LSTMDualEncoder, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        #print (n_vocab)
        if pretrained_emb is not None:
             self.word_embed.weight.data.copy_(pretrained_emb)

        #self.word_embed.weight.data = position_encoding_init(n_vocab, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))

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
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.word_embed(x1)
        x2_emb = self.word_embed(x2)


        # Each is (1 x batch_size x h_dim)
        _, (c, _) = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        return c.squeeze(), r.squeeze()

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


class LSTMDualEncPack(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False):
        super(LSTMDualEncPack, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        #print (n_vocab)
        if pretrained_emb is not None:
             self.word_embed.weight.data.copy_(pretrained_emb)

        #self.word_embed.weight.data = position_encoding_init(n_vocab, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))

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

    def forward(self, x1, x1_l, x2, x2_l):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        x1_l, x1_p_idx = x1_l.sort(0, descending=True)
        x2_l, x2_p_idx = x2_l.sort(0, descending=True)
        x1 = x1[x1_p_idx]
        x2 = x2[x2_p_idx]
        c, r = self.forward_enc(x1, x1_l, x2, x2_l)
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x1_l, x2, x2_l):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.word_embed(x1)
        x2_emb = self.word_embed(x2)
        x1_pack = pack_padded_sequence(x1_emb, x1_l.cpu().numpy(), batch_first=True)
        x2_pack = pack_padded_sequence(x2_emb, x2_l.cpu().numpy(), batch_first=True)
        #print (x1_pack)
        # Each is (1 x batch_size x h_dim)
        pack_c, (h, _) = self.rnn(x1_pack)
        pack_r, (h, _) = self.rnn(x2_pack)
        c, _ = pad_packed_sequence(pack_c, batch_first=True)
        r, _ = pad_packed_sequence(pack_r, batch_first=True)
        #return c.squeeze(), r.squeeze()
        return c[:, -1], r[:, -1]

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


class LSTMDualEncoderDeep(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False, max_seq_len=160, emb_drop=0.6):
        super(LSTMDualEncoderDeep, self).__init__()
        print (n_vocab)
        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=1)
        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        #self.word_embed.weight.data = position_encoding_init(n_vocab, emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.dropout_p = emb_drop
        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.dropout = nn.Dropout(self.dropout_p)
        self.max_seq_len = max_seq_len
        #self.attn = SelfAttention(h_dim, batch_first=True)
        self.init_params_()
        #self.attn = Attention(h_dim, h_dim)
        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

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
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.word_embed(x1)
        x1_emb = self.dropout(x1_emb)
        x2_emb = self.word_embed(x2)
        x2_emb = self.dropout(x2_emb)
        #seq_lens = (self.max_seq_len * x1_emb.size(0))

        #packed_seq_x1 = pack_padded_sequence(x1_emb, lengths=seq_lens, batch_first=True)
        #packed_seq_x2 = pack_padded_sequence(x2_emb, lengths=seq_lens, batch_first=True)

        # Each is (1 x batch_size x h_dim)
        _, (c, _) = self.rnn(x1_emb)

        #wattn, attn_mask = self.attn(c[0][-1], c[1][-1])
        _, (r, _) = self.rnn(x2_emb)

        return c.squeeze(), r.squeeze()

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


class EmbMM(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False, max_seq_len=160, emb_drop=0.5):
        super(EmbMM, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, sparse=False, padding_idx=0)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        #self.word_embed = torch.autograd.Variable(self.word_embed)
        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.h_dim = h_dim
        self.dropout_p = emb_drop
        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.dropout = nn.Dropout(self.dropout_p)
        self.max_seq_len = max_seq_len
        self.out_h = nn.Linear(h_dim, 1)
        self.out_drop = nn.Dropout(0.2)
        self.nl = nn.Tanh()
        #self.maxpool = nn.MaxPool1d(128, stride=1)
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

        o = self.forward_fc(c, r)
        #print (o.size())
        #f_o = self.out_h(o)
        return o.view(-1)
        #return f_o.squeeze()

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)

        x1_emb = self.word_embed(x1)
        x1_emb = self.dropout(x1_emb)
        x2_emb = self.word_embed(x2)
        x2_emb = self.dropout(x2_emb)



        # Each is (1 x batch_size x h_dim)
        _, (c, _) = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        return c.squeeze(), r.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        results = []

        # (batch_size x 1 x h_dim)
        for i in range(len(c)):
            #print (c[i].size())
            context_h = c[i].view(1, self.h_dim)
            #context_h = c[i]
            #response_h = r[i].view(self.h_dim, 1)
            response_h = r[i]
            w_mm = torch.mm(context_h, self.M).squeeze()
            #print (w_mm.size())
            ans = self.nl(w_mm * response_h)
            #print (ans.size())
            results.append(self.out_drop(self.out_h(ans)))

        o = torch.stack(results)
        #print (o.size())
        return o.squeeze()


class CrossConvNet(nn.Module):

    def __init__(self, emb_dim, n_vocab, max_seq_len, pretrained_emb=None, k=1, gpu=False):
        super(CrossConvNet, self).__init__()

        self.n_vocab = n_vocab
        self.emb_dim = emb_dim
        self.k = k
        self.max_seq_len = max_seq_len

        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.fc1 = nn.Linear(max_seq_len*k, 1)
        self.fc2 = nn.Linear(max_seq_len*k, 1)

        if gpu:
            self.cuda()

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        h: vector of (batch_size)
        """
        # Both are (batch_size, emb_dim, seq_len)
        x1_emb = self.word_embed(x1).transpose(1, 2)
        x2_emb = self.word_embed(x2).transpose(1, 2)

        # Pad into (batch_size, emb_dim, max_seq_len)
        x1_seq_len = x1_emb.size(-1)
        x2_seq_len = x2_emb.size(-1)

        pad_size1 = self.max_seq_len - x1_seq_len
        pad_size2 = self.max_seq_len - x2_seq_len

        x1_emb = F.pad(x1_emb, (0, pad_size1))
        x2_emb = F.pad(x2_emb, (0, pad_size2))

        # Take dot product. S is (batch_size, L, L)
        a = torch.bmm(x2_emb.transpose(1, 2), x1_emb)
        # k-maxpool: (batch_size, L, L) -> (batch_size, k, L)
        a, _ = torch.topk(a, k=self.k, dim=1, sorted=False)
        # Ravel: (batch_size, k, L) -> (batch_size, k*L)
        a = a.view(-1, self.k*self.max_seq_len)

        # batch_size x 1
        o1 = self.fc1(a)
        o2 = self.fc2(a)

        return o1.view(-1), o2.view(-1)


class CCN_LSTM(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, max_seq_len=160, k=1, pretrained_emb=None, gpu=False):
        super(CCN_LSTM, self).__init__()

        self.lstm = LSTMDualEncoder(emb_dim, n_vocab, h_dim, pretrained_emb, gpu)
        self.ccn = CrossConvNet(emb_dim, n_vocab, max_seq_len, pretrained_emb, k, gpu)

        self.fc = nn.Linear(3, 1)

        if gpu:
            self.cuda()

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        o_lstm = self.lstm(x1, x2).unsqueeze(1)
        o1_ccn, o2_ccn = self.ccn(x1, x2)

        inputs = torch.cat([o_lstm, o1_ccn.unsqueeze(1), o2_ccn.unsqueeze(1)], 1)
        o = self.fc(inputs)

        return o.view(-1)


class AttnLSTMDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, max_seq_len=160, conv=False, gpu=False):
        super(AttnLSTMDualEncoder, self).__init__()

        self.conv = conv
        self.max_seq_len = 160
        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        if not conv:
            self.attention = nn.Sequential(
                nn.Linear(max_seq_len*h_dim, max_seq_len),
                nn.Softmax(dim=1)
            )
        else:
            self.conv3 = nn.Conv2d(1, 100, (3, h_dim))
            self.conv4 = nn.Conv2d(1, 100, (4, h_dim))
            self.conv5 = nn.Conv2d(1, 100, (5, h_dim))
            self.attention = nn.Sequential(
                nn.Linear(300, max_seq_len),
                nn.Softmax(dim=1)
            )

        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))

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
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.word_embed(x1)
        x2_emb = self.word_embed(x2)

        # Each is (1 x batch_size x h_dim)
        c_hs, _ = self.rnn(x1_emb)
        _, (r, _) = self.rnn(x2_emb)

        # Attention weights
        if self.conv:
            c_a = self.attention(self.forward_conv(c_hs))
            #r_a = self.attention(self.forward_conv(r_hs))
        else:

            c_a = self.attention(torch.cat(c_hs.contiguous().view(x1.size(0), -1), x1_emb))
            #r_a = self.attention(r_hs.contiguous().view(x2.size(0), -1))

        # Apply attention to RNN outputs
        c = torch.bmm(c_hs.transpose(1, 2), c_a.unsqueeze(2))
        #r = torch.bmm(r_hs.transpose(1, 2), r_a.unsqueeze(2))

        return c.squeeze(), r.squeeze()

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

    def forward_conv(self, inputs):
        x = inputs.unsqueeze(1)

        x3 = F.relu(self.conv3(x)).squeeze()
        x4 = F.relu(self.conv4(x)).squeeze()
        x5 = F.relu(self.conv5(x)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        return x


class AttnLstmDeep(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, max_seq_len=160, conv=False, gpu=False):
        super(AttnLstmDeep, self).__init__()

        self.conv = conv
        self.max_seq_len = 160
        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=1)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        if not conv:
            self.attention = nn.Sequential(
                nn.Linear(max_seq_len*h_dim, max_seq_len),
                nn.Softmax(dim=1)
            )
        else:
            self.conv3 = nn.Conv2d(1, 100, (3, h_dim))
            self.conv4 = nn.Conv2d(1, 100, (4, h_dim))
            self.conv5 = nn.Conv2d(1, 100, (5, h_dim))
            self.attention = nn.Sequential(
                nn.Linear(300, max_seq_len),
                nn.Softmax(dim=1)
            )

        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))

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
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.word_embed(x1)
        x2_emb = self.word_embed(x2)

        # Each is (1 x batch_size x h_dim)
        c_hs, _ = self.rnn(x1_emb)
        r_hs, _ = self.rnn(x2_emb)

        # Attention weights
        if self.conv:
            c_a = self.attention(self.forward_conv(c_hs))
            r_a = self.attention(self.forward_conv(r_hs))
        else:
            c_a = self.attention(c_hs.contiguous().view(x1.size(0), -1))
            r_a = self.attention(r_hs.contiguous().view(x2.size(0), -1))

        # Apply attention to RNN outputs
        c = torch.bmm(c_hs.transpose(1, 2), c_a.unsqueeze(2))
        r = torch.bmm(r_hs.transpose(1, 2), r_a.unsqueeze(2))

        return c.squeeze(), r.squeeze()

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

    def forward_conv(self, inputs):
        x = inputs.unsqueeze(1)

        x3 = F.relu(self.conv3(x)).squeeze()
        x4 = F.relu(self.conv4(x)).squeeze()
        x5 = F.relu(self.conv5(x)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        return x
