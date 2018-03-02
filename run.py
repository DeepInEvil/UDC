import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from model import CNNDualEncoder
from data import UDC
from eval import accuracy

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


dataset = UDC(embed_dim=args.emb_dim, batch_size=args.mb_size, gpu=args.gpu)
model = CNNDualEncoder(dataset.embed_dim, dataset.vocab_size, dataset.vectors)

solver = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(args.n_epoch):
    print('\nEpoch-{}'.format(epoch))
    print('-------------------------------------------')

    for it, mb in enumerate(dataset.train_iter()):
        output = model(mb.context.t(), mb.response.t())
        loss = F.binary_cross_entropy_with_logits(output, mb.label)

        loss.backward()
        solver.step()
        solver.zero_grad()

        if it % 100 == 0:
            total_acc = 0
            n = 0

            for mb in dataset.valid_iter():
                y_pred = F.sigmoid(model(mb.context.t(), mb.response.t()))
                total_acc += accuracy(y_pred, mb.label, mean=False)
                n += mb.batch_size

            val_acc = total_acc / n

            print('Iter-{}; loss: {:.3f}; val_acc: {:.3f}'
                  .format(it, loss.data[0], val_acc.data[0]))
