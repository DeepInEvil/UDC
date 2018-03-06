import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from model import CNNDualEncoder
from data import UDC
from evaluation import recall_at_k

import argparse


parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--emb_dim', type=int, default=100, metavar='',
                    help='embedding dimension (default: 100)')
parser.add_argument('--mb_size', type=int, default=32, metavar='',
                    help='size of minibatch (default: 32)')
parser.add_argument('--n_epoch', type=int, default=10, metavar='',
                    help='number of iterations (default: 10)')

args = parser.parse_args()


dataset = UDC(
    train_file='train10k.csv', valid_file='valid500.csv', test_file='test500.csv',
    embed_dim=args.emb_dim, batch_size=args.mb_size, gpu=args.gpu
)
model = CNNDualEncoder(dataset.embed_dim, dataset.vocab_size, dataset.vectors, args.gpu)

solver = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(args.n_epoch):
    print('\nEpoch-{}'.format(epoch))
    print('-------------------------------------------')

    for it, mb in enumerate(dataset.train_iter()):
        model.train()

        output = model(mb.context.t(), mb.response.t())
        loss = F.binary_cross_entropy_with_logits(output, mb.label)

        loss.backward()
        solver.step()
        solver.zero_grad()

        if it % 100 == 0:
            model.eval()
            scores = []

            for mb in dataset.valid_iter():
                context = mb.context.t()

                # Get score for positive/ground-truth response
                score_pos = model(context, mb.positive.t()).unsqueeze(1)
                # Get scores for negative samples
                score_negs = [
                    model(context, getattr(mb, 'negative_{}'.format(i)).t()).unsqueeze(1)
                    for i in range(1, 10)
                ]
                # Total scores, positives at position zero
                scores_mb = torch.cat([score_pos, *score_negs], dim=1)

                scores.append(scores_mb)

            scores = torch.cat(scores, dim=0)
            recall_at_ks = [
                r.cpu().data[0] if args.gpu else r.data[0]
                for r in recall_at_k(scores)
            ]

            print('Iter-{}; loss: {:.3f}; recall@1: {:.3f}; recall@3: {:.3f}; recall@5: {:.3f}'
                  .format(it, loss.data[0], recall_at_ks[0], recall_at_ks[2], recall_at_ks[4]))
