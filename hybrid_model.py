import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import LSTMDualEncoder


class HybridModel(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim, z_dim, pretrained_emb=None, eou_idx=0, gpu=False):
        super(HybridModel, self).__init__()

        self.EOU_IDX = eou_idx
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gpu = gpu

        self.retrieval_model = LSTMDualEncoder(emb_dim, n_vocab, h_dim, pretrained_emb)
        self.latent_fc = nn.Linear(h_dim+h_dim, z_dim)
        self.decoder = nn.LSTM(
            input_size=emb_dim, hidden_size=z_dim,
            num_layers=1, batch_first=True
        )
        self.decoder_fc = nn.Linear(z_dim, n_vocab)

        if gpu:
            self.cuda()

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        out_retrieval: vector of (batch_size)
        out_generative: matrix of (batch_size*seq_len, n_vocab)
        """
        # Retrieval model
        c, r = self.retrieval_model.forward_enc(x1, x2)
        out_retrieval = self.retrieval_model.forward_fc(c, r).view(-1)

        # Generative model
        z = self.latent_fc(torch.cat([c, r], 1)).unsqueeze(0)
        init_state = (z, z)

        mb_size = x2.size(0)

        # sentence: 'I want to fly __eou__'
        # decoder input : '__eou__ I want to fly'
        # decoder target: 'I want to fly __eou__'
        dec_inputs = x2[:, :-1]

        eous = Variable(torch.LongTensor([self.EOU_IDX])).repeat(mb_size, 1)
        eous = eous.cuda() if self.gpu else eous

        dec_inputs = torch.cat([eous, dec_inputs], 1)

        # Forward decoder
        dec_inputs = self.retrieval_model.word_embed(dec_inputs)
        outputs, _ = self.decoder(dec_inputs, init_state)
        outputs = outputs.contiguous().view(-1, self.z_dim)
        out_generative = self.decoder_fc(outputs)

        return out_retrieval, out_generative


class VariationalHybridModel(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim, z_dim, pretrained_emb=None, eou_idx=0, gpu=False):
        super(VariationalHybridModel, self).__init__()

        self.EOU_IDX = eou_idx
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gpu = gpu

        self.retrieval_model = LSTMDualEncoder(emb_dim, n_vocab, h_dim, pretrained_emb)
        self.latent_mu_fc = nn.Linear(h_dim+h_dim, z_dim)
        self.latent_logvar_fc = nn.Linear(h_dim+h_dim, z_dim)
        self.decoder = nn.LSTM(
            input_size=emb_dim, hidden_size=z_dim,
            num_layers=1, batch_first=True
        )
        self.decoder_fc = nn.Linear(z_dim, n_vocab)

        if gpu:
            self.cuda()

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        out_retrieval: vector of (batch_size)
        out_generative: matrix of (batch_size*seq_len, n_vocab)
        """
        mb_size = x2.size(0)

        # Retrieval model
        c, r = self.retrieval_model.forward_enc(x1, x2)
        out_retrieval = self.retrieval_model.forward_fc(c, r).view(-1)

        # Generative model
        z_mu = self.latent_mu_fc(torch.cat([c, r], 1))
        z_logvar = self.latent_logvar_fc(torch.cat([c, r], 1))

        # Reparam. trick
        eps = Variable(torch.randn(mb_size, self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        z = z_mu + torch.exp(z_logvar/2) * eps
        z = z.unsqueeze(0)

        init_state = (z, z)

        # sentence: 'I want to fly __eou__'
        # decoder input : '__eou__ I want to fly'
        # decoder target: 'I want to fly __eou__'
        dec_inputs = x2[:, :-1]

        eous = Variable(torch.LongTensor([self.EOU_IDX])).repeat(mb_size, 1)
        eous = eous.cuda() if self.gpu else eous

        dec_inputs = torch.cat([eous, dec_inputs], 1)

        # Forward decoder
        dec_inputs = self.retrieval_model.word_embed(dec_inputs)
        outputs, _ = self.decoder(dec_inputs, init_state)
        outputs = outputs.contiguous().view(-1, self.z_dim)
        out_generative = self.decoder_fc(outputs)

        return out_retrieval, out_generative, z_mu, z_logvar
