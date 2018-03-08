import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from hybrid_model import VariationalHybridModel
from data import UDC
from evaluation import recall_at_k
from util import save_model

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
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--toy_data', default=False, action='store_true',
                    help='whether to use toy dataset (10k instead of 1m)')

args = parser.parse_args()

max_seq_len = 160
k = 1
h_dim = 256
z_dim = 256

if args.toy_data:
    dataset = UDC(
        train_file='train10k.csv', valid_file='valid500.csv', test_file='test500.csv',
        embed_dim=args.emb_dim, batch_size=args.mb_size, max_seq_len=max_seq_len, gpu=args.gpu
    )
else:
    dataset = UDC(
        train_file='train.csv', valid_file='valid.csv', test_file='test.csv',
        embed_dim=args.emb_dim, batch_size=args.mb_size, max_seq_len=max_seq_len, gpu=args.gpu
    )

eou_idx = dataset.TEXT.vocab.stoi['__eou__']
model = VariationalHybridModel(dataset.embed_dim, dataset.vocab_size, h_dim, z_dim, dataset.vectors, eou_idx, args.gpu)

solver = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(args.n_epoch):
    print('\nEpoch-{}'.format(epoch))
    print('-------------------------------------------')

    for it, mb in enumerate(dataset.train_iter()):
        model.train()

        out_retrieval, out_generative, z_mu, z_logvar = model(mb.context, mb.response)

        loss_ret = F.binary_cross_entropy_with_logits(out_retrieval, mb.label)

        loss_recon = F.cross_entropy(out_generative, mb.response.view(-1))
        loss_kl = torch.mean(0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1 - z_logvar, 1))
        loss_vae = loss_recon + loss_kl

        loss = loss_ret + loss_vae

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 10)
        solver.step()
        solver.zero_grad()

        if it % 99999 == 0:
            model.eval()
            scores = []

            for mb in dataset.valid_iter():
                # Get score for positive/ground-truth response
                score_pos = model(mb.context, mb.positive)[0].unsqueeze(1)
                # Get scores for negative samples
                score_negs = [
                    model(mb.context, getattr(mb, 'negative_{}'.format(i)))[0].unsqueeze(1)
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

    save_model(model, 'hybrid_variational')
