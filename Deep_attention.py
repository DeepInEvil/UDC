import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size),
                                     requires_grad=True)

        init.xavier_uniform(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, inputs):

        if isinstance(inputs, nn.utils.rnn.PackedSequence):
            # unpack output
            inputs, lengths = nn.utils.rnn.pad_packed_sequence(inputs,
                                                  batch_first=self.batch_first)
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            # (batch_size, hidden_size, 1)
                            )

        attentions = F.softmax(F.relu(weights.squeeze()))

        # create mask based on the sentence lengths
        mask = Variable(torch.ones(attentions.size())).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).expand_as(attentions)  # sums per row
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions