import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UtteranceEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, gpu=True):
        super(UtteranceEncoder, self).__init__()

        self.embedding = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, h_dim)

        if gpu:
            self.cuda()

    def forward(self, input):
        emb = self.embedding(input)
        _, h = self.gru(emb)
        return h


class ContextEncoder(nn.Module):

    def __init__(self, h_dim=200, gpu=True):
        super(ContextEncoder, self).__init__()
        self.gru = nn.GRU(h_dim, h_dim)

        if gpu:
            self.cuda()

    def forward(self, input, h_prev):
        _, h = self.gru(input, h_prev)
        return h


class ResponseAttnDecoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, emb_drop=0.2, gpu=True):
        super(ResponseAttnDecoder, self).__init__()

        self.n_vocab = n_vocab
        self.h_dim = h_dim

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_drop)

        self.gru = nn.GRU(emb_dim+h_dim, h_dim)
        self.decoder_fc = nn.Linear(h_dim, n_vocab)

        if gpu:
            self.cuda()

    def forward(self, input, context):
        emb = self.emb_drop(self.word_embed(input))
        input = torch.cat((emb, context), 2)

        out = self.gru(input)  # (seq_len x batch_size x h_dim)
        out = self.decoder_fc(out.view(-1, self.h_dim))  # (seq_len*batch_size, n_vocab)

        return out


class HRED(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, emb_drop=0.2, start_token=1, gpu=True):
        super(HRED, self).__init__()

        self.START = start_token
        self.gpu = gpu

        self.utterance_enc = UtteranceEncoder(emb_dim, n_vocab, h_dim, gpu)
        self.context_enc = ContextEncoder(h_dim, gpu)
        self.response_dec = ResponseAttnDecoder(emb_dim, n_vocab, h_dim, emb_drop, gpu)

    def step(self, input, target, ctx_h_prev):
        """
        Single step of HRED, i.e. process single utterance in single dialogue.

        input: LongTensor of (seq_len)
        target: LongTensor of (seq_len)
        """
        # Encode utterance
        enc_h = self.utterance_enc(input)

        # Update context
        ctx_h = self.context_enc(enc_h, ctx_h_prev)

        # Learn decoder
        dec_input = self._get_dec_input(target)
        dec_h = ctx_h

        out, _ = self.response_dec(dec_input, ctx_h)  # (seq_len x mb_size, h)

        return out, ctx_h

    def forward(self, inputs):
        """
        Training step for given full dialogues.

        inputs: list of T sequences where T is number of turns
        """
        mb_size = inputs.size(2)
        total_loss = 0
        ctx_h = None

        for t in range(len(inputs)-1):
            input = inputs[t]  # (max_seq_len, mb_size)
            target = inputs[t+1]  # (max_seq_len, mb_size)

            out, ctx_h = self.step(input, target, ctx_h)

            loss = F.cross_entropy(out, target.view(-1))
            total_loss += loss

        return total_loss / mb_size

    def _get_dec_input(self, dec_target):
        # sentence: 'I want to fly __eot__'
        # decoder input : '__eot__ I want to fly'
        # decoder target: 'I want to fly __eot__'
        dec_input = dec_target[:-1, :]
        mb_size = dec_input.size(1)

        start = Variable(torch.LongTensor([self.START])).repeat(1, mb_size)
        start = start.cuda() if self.gpu else start

        dec_input = torch.cat([start, dec_input], 0)

        return dec_input
