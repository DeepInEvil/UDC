import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import LSTMDualEncoder, GRUDualEncoder, GRUDualEncoderAttn
from itertools import chain

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


class VAEDualEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim, z_dim, pretrained_emb=None, eos_idx=63346, pad_idx=0, emb_drop=0.5, gpu=False):
        super(VAEDualEncoder, self).__init__()

        self.EOS_IDX = eos_idx
        self.PAD_IDX = pad_idx
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gpu = gpu

        self.retrieval_model = GRUDualEncoder(emb_dim, n_vocab, h_dim, pretrained_emb, gpu, emb_drop, pad_idx)
        self.latent_mu_fc = nn.Linear(h_dim, z_dim)
        self.latent_logvar_fc = nn.Linear(h_dim, z_dim)

        self.dropout = nn.Dropout(p=emb_drop)
        self.decoder = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True
        )
        self.decoder_fc = nn.Linear(h_dim, n_vocab)

        if gpu:
            self.cuda()

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        z = mu + torch.exp(logvar/2) * eps
        return z.unsqueeze(0)

    def forward_decoder(self, x, init_state, max_seq_len=16):
        # sentence: 'I want to fly __eos__ __pad__ __pad__'
        # decoder input : '__eos__ I want to fly __eos__ __pad__'
        # decoder target: 'I want to fly __eos__ __pad__ __pad__'
        inputs = x[:, :max_seq_len-1]
        eous = Variable(torch.LongTensor([self.EOS_IDX])).repeat(x.size(0), 1)
        eous = eous.cuda() if self.gpu else eous
        inputs = torch.cat([eous, inputs], 1)

        targets = x[:, :max_seq_len]

        # Forward decoder
        inputs = self.dropout(self.retrieval_model.word_embed(inputs))
        outputs, _ = self.decoder(inputs, init_state)
        outputs = outputs.contiguous().view(-1, self.h_dim)
        outputs = self.decoder_fc(outputs)  # (mbsize*seq_len) x n_vocab

        return outputs, targets

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        out: vector of (batch_size)
        """
        mb_size = x2.size(0)

        c, r = self.retrieval_model.forward_enc(x1, x2)

        zc_mu = self.latent_mu_fc(c)
        zc_logvar = self.latent_logvar_fc(c)
        zc = self.sample_z(zc_mu, zc_logvar)
        outputs_c, targets_c = self.forward_decoder(x1, zc)

        recon_loss_c = F.cross_entropy(outputs_c, targets_c.contiguous().view(-1))
        kl_loss_c = torch.mean(0.5 * torch.sum(torch.exp(zc_logvar) + zc_mu**2 - 1 - zc_logvar, 1))

        zr_mu = self.latent_mu_fc(r)
        zr_logvar = self.latent_logvar_fc(r)
        zr = self.sample_z(zr_mu, zr_logvar)
        outputs_r, targets_r = self.forward_decoder(x2, zr)

        # print(outputs_c.size(), outputs_r.size(), outputs_retrieval.size())
        recon_loss_r = F.cross_entropy(outputs_r, targets_r.contiguous().view(-1))
        kl_loss_r = torch.mean(0.5 * torch.sum(torch.exp(zr_logvar) + zr_mu**2 - 1 - zr_logvar, 1))

        # outputs_retrieval = self.retrieval_model.forward_fc(zc_mu, zr_mu).squeeze()
        outputs_retrieval = self.retrieval_model.forward_fc(c, r).squeeze()

        return outputs_retrieval, recon_loss_c, kl_loss_c, recon_loss_r, kl_loss_r

    def predict(self, x1, x2):
        c, r = self.retrieval_model.forward_enc(x1, x2)
        outputs_retrieval = self.retrieval_model.forward_fc(c, r).squeeze()

        return outputs_retrieval


class GRUDualEncoderPlusVAE(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim, z_dim, pretrained_emb=None, eos_idx=63346, pad_idx=0, emb_drop=0.5, gpu=False):
        super(GRUDualEncoderPlusVAE, self).__init__()

        self.EOS_IDX = eos_idx
        self.PAD_IDX = pad_idx
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.gpu = gpu

        self.retrieval_model = GRUDualEncoderAttn(emb_dim, n_vocab, h_dim, pretrained_emb, gpu, emb_drop, pad_idx, z_dim=z_dim)
        self.dropout = nn.Dropout(p=emb_drop)

        self.vae_encoder = nn.GRU(emb_dim, h_dim, batch_first=True)
        self.latent_mu_fc = nn.Linear(h_dim, z_dim)
        self.latent_logvar_fc = nn.Linear(h_dim, z_dim)

        self.vae_decoder = nn.GRU(emb_dim, h_dim, batch_first=True)
        self.vae_decoder_fc = nn.Linear(h_dim, n_vocab)

        # Grouping params
        self.retrieval_params = self.retrieval_model.parameters()
        self.vae_params = chain(
            self.retrieval_model.word_embed.parameters(),
            self.vae_encoder.parameters(),
            self.latent_mu_fc.parameters(),
            self.latent_logvar_fc.parameters(),
            self.vae_decoder.parameters(),
            self.vae_decoder_fc.parameters()
        )

        if gpu:
            self.cuda()

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), self.z_dim))
        eps = eps.cuda() if self.gpu else eps
        z = mu + torch.exp(logvar/2) * eps
        return z.unsqueeze(0)

    def forward_decoder(self, x, init_state, max_seq_len=20):
        # sentence: 'I want to fly __eos__ __pad__ __pad__'
        # decoder input : '__eos__ I want to fly __eos__ __pad__'
        # decoder target: 'I want to fly __eos__ __pad__ __pad__'
        inputs = x[:, :max_seq_len-1]
        eous = Variable(torch.LongTensor([self.EOS_IDX])).repeat(x.size(0), 1)
        eous = eous.cuda() if self.gpu else eous
        inputs = torch.cat([eous, inputs], 1)

        targets = x[:, :max_seq_len]

        # Forward decoder
        inputs = self.dropout(self.retrieval_model.word_embed(inputs))
        outputs, _ = self.vae_decoder(inputs, init_state)
        outputs = outputs.contiguous().view(-1, self.h_dim)
        outputs = self.vae_decoder_fc(outputs)  # (mbsize*seq_len) x n_vocab

        return outputs, targets

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        out: vector of (batch_size)
        """
        mb_size = x2.size(0)

        c, r = self.retrieval_model.forward_enc(x1, x2)

        vae_c = self.vae_encoder(self.retrieval_model.word_embed(x1))[1].squeeze()
        zc_mu = self.latent_mu_fc(vae_c)

        vae_r = self.vae_encoder(self.retrieval_model.word_embed(x2))[1].squeeze()
        zr_mu = self.latent_mu_fc(vae_r)

        c = torch.cat([c, zc_mu], -1)
        r = torch.cat([r, zr_mu], -1)

        out = self.retrieval_model.forward(c, r).squeeze()

        return out

    def forward_vae(self, x):
        enc_out = self.vae_encoder(self.retrieval_model.word_embed(x))[1].squeeze()
        z_mu = self.latent_mu_fc(enc_out)
        z_logvar = self.latent_logvar_fc(enc_out)
        z = self.sample_z(z_mu, z_logvar)
        outputs, targets = self.forward_decoder(x, z)

        recon_loss = F.cross_entropy(outputs, targets.contiguous().view(-1))
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1 - z_logvar, 1))

        return recon_loss, kl_loss

    def predict(self, x1, x2):
        return self.forward(x1, x2)
