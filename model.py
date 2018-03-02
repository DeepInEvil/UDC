import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, pretrained_emb=None):
        super(CNNDualEncoder, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.conv3 = nn.Conv2d(1, 100, (3, emb_dim))
        self.conv4 = nn.Conv2d(1, 100, (4, emb_dim))
        self.conv5 = nn.Conv2d(1, 100, (5, emb_dim))
        self.bilinear = nn.Bilinear(300, 300, 1)

    def forward(self, x1, x2):
        """
        x1, x2: tensors of (batch_size, seq_len)
        """
        emb_x1 = self.word_embed(x1)
        emb_x2 = self.word_embed(x2)

        c1 = self._forward(emb_x1)
        c2 = self._forward(emb_x1)

        o = self.bilinear(c1, c2)

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
