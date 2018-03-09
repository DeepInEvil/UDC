import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM
from data import UDC
from evaluation import recall_at_k, eval_model
from util import save_model

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--emb_dim', type=int, default=100, metavar='',
                    help='embedding dimension (default: 100)')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--mb_size', type=int, default=128, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--toy_data', default=False, action='store_true',
                    help='whether to use toy dataset (10k instead of 1m)')

args = parser.parse_args()

max_seq_len = 160
k = 1
h_dim = args.h_dim

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

# model = CNNDualEncoder(dataset.embed_dim, dataset.vocab_size, dataset.vectors, args.gpu)
model = LSTMDualEncoder(dataset.embed_dim, dataset.vocab_size, h_dim, dataset.vectors, args.gpu)
# model = CCN_LSTM(dataset.embed_dim, dataset.vocab_size, h_dim, max_seq_len, k, dataset.vectors, args.gpu)

solver = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.n_epoch):
    print('\n\n-------------------------------------------')
    print('Epoch-{}'.format(epoch))
    print('-------------------------------------------')

    model.train()

    train_iter = tqdm(enumerate(dataset.train_iter()))
    train_iter.set_description_str('Training')

    for it, mb in train_iter:
        output = model(mb.context, mb.response)
        loss = F.binary_cross_entropy_with_logits(output, mb.label)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 10)
        solver.step()
        solver.zero_grad()

    # Validation
    recall_at_ks = eval_model(model, dataset, args.gpu)

    print('\nLoss: {:.3f}; recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(loss.data[0], recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))

    save_model(model, 'ccn_lstm')
