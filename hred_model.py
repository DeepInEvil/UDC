import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class UtteranceEncoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, gpu=True):
        super(UtteranceEncoder, self).__init__()

        self.embedding = nn.Embedding(n_vocab, emb_dim)
        self.gru = nn.GRU(emb_dim, h_dim)

        if gpu:
            self.cuda()

    def forward(self, input):
        emb = self.embedding(input).unsqueeze(1)
        _, h = self.gru(emb)
        return h


class ContextEncoder(nn.Module):

    def __init__(self, h_dim=200, gpu=True):
        super(ContextEncoder, self).__init__()
        self.gru = nn.GRU(h_dim, h_dim)

        if gpu:
            self.cuda()

    def forward(self, input, h_prev):
        _, h = self.gru(input.view(1, 1, -1), h_prev)
        return h


class ResponseAttnDecoder(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, emb_drop=0.2, max_seq_len=50, gpu=True):
        super(ResponseAttnDecoder, self).__init__()

        self.max_seq_len = max_seq_len

        self.word_embed = nn.Embedding(n_vocab, emb_dim)
        self.emb_drop = nn.Dropout(emb_drop)

        self.gru = nn.GRU(emb_dim+h_dim, h_dim)
        self.decoder_fc = nn.Linear(h_dim, n_vocab)

        if gpu:
            self.cuda()

    def forward(self, input, context, h_prev):
        emb = self.emb_drop(self.word_embed(input)).view(1, 1, -1)
        input = torch.cat((emb, context), 2)

        out, h = self.gru(input, h_prev)
        out = self.decoder_fc(out[0])

        return out, h


class HRED(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=200, emb_drop=0.2, max_seq_len=50, start_token=1, gpu=True):
        super(HRED, self).__init__()

        self.START = start_token
        self.gpu = gpu

        self.utterance_enc = UtteranceEncoder(emb_dim, n_vocab, h_dim, gpu)
        self.context_enc = ContextEncoder(h_dim, gpu)
        self.response_dec = ResponseAttnDecoder(emb_dim, n_vocab, h_dim, emb_drop, max_seq_len, gpu)

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

        # Decode response
        inp = Variable(torch.LongTensor([[self.START]]))
        inp = inp.cuda() if self.gpu else inp

        dec_h = ctx_h
        loss = 0

        # Always use teacher forcing
        for i in range(len(target)):
            out, dec_h = self.response_dec(inp, ctx_h, dec_h)
            loss += F.cross_entropy(out, target[i], size_average=False)
            inp = target[i]

        return loss, ctx_h

    def forward(self, inputs):
        """
        Training step for given full dialogues.

        inputs: list of T sequences where T is number of turns
        """
        total_loss = 0
        ctx_h = None

        for i in range(len(inputs)-1):
            input = inputs[i]
            target = inputs[i+1]
            loss, ctx_h = self.step(input, target, ctx_h)
            total_loss += loss

        return total_loss
