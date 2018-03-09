import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, pretrained_emb=None, gpu=False):
        super(CNNDualEncoder, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.conv3 = nn.Conv2d(1, 100, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, emb_dim))
        self.M = nn.Parameter(nn.init.xavier_normal(torch.FloatTensor(300, 300)))

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
        # (batch_size x 300 x 1)
        c2 = self._forward(emb_x2).unsqueeze(2)

        # (batch_size x 1 x 300)
        o = torch.mm(c1, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, c2)

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

        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.LSTM(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )

        self.M = nn.Parameter(nn.init.xavier_normal(torch.FloatTensor(h_dim, h_dim)))

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

        return o


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

        self.fc = nn.Linear(max_seq_len*k, 1)

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
        h = self.fc(a)

        return h.view(-1)


class CCN_LSTM(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, max_seq_len=160, k=1, pretrained_emb=None, gpu=False):
        super(CCN_LSTM, self).__init__()

        self.lstm = LSTMDualEncoder(emb_dim, n_vocab, h_dim, pretrained_emb, gpu)
        self.ccn = CrossConvNet(emb_dim, n_vocab, max_seq_len, pretrained_emb, k, gpu)

        self.fc = nn.Linear(2, 1)

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
        o_ccn = self.ccn(x1, x2).unsqueeze(1)

        inputs = torch.cat([o_lstm, o_ccn], 1)
        o = self.fc(inputs)

        return o.view(-1)
