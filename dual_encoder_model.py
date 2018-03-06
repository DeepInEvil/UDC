import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 num_directions,
                 dropout,
                 bidirectional,
                 rnn_type,
                 p_dropout,
                 pretrained_emb=None):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.bidirectional = False
        self.rnn_type = 'lstm'
        self.p_dropout = p_dropout

        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout=dropout,
                            bidirectional=False)
        self.dropout_layer = nn.Dropout(self.p_dropout)

        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        self.init_weights()

    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a=-0.25, b=0.25)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        #outputs, hiddens_tuple = self.lstm(embeddings)

        outputs, hiddens_tuple = self.lstm(embeddings)

        last_hidden = outputs[-1]  # access last lstm layer, dimensions: (batch_size x hidden_size)
        last_hidden = self.dropout_layer(last_hidden)  # dimensions: (batch_size x hidden_size)

        return last_hidden


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, context_tensor, response_tensor):
        context_last_hidden = self.encoder(context_tensor)  # dimensions: (batch_size x hidden_size)
        response_last_hidden = self.encoder(response_tensor)  # dimensions: (batch_size x hidden_size)

        context = context_last_hidden.mm(self.M).cuda()
        #context = context_last_hidden.mm(self.M)  # dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size)  # dimensions: (batch_size x 1 x hidden_size)

        response = response_last_hidden.view(-1, self.hidden_size, 1)  # dimensions: (batch_size x hidden_size x 1)

        score = torch.bmm(context, response).view(-1, 1).cuda()
        #score = torch.bmm(context, response).view(-1, 1)  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        return score