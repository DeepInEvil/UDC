import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from hred_model import HRED
from data import UDCv1
from evaluation import eval_model_v1
from util import save_model, clip_gradient_threshold

import argparse
import pickle
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--emb_drop', type=float, default=0.3, metavar='',
                    help='embedding dropout (default: 0.3)')
parser.add_argument('--mb_size', type=int, default=256, metavar='',
                    help='size of minibatch (default: 256)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--randseed', type=int, default=123, metavar='',
                    help='random seed (default: 123)')
parser.add_argument('--no_tqdm', default=False, action='store_true',
                    help='disable tqdm progress bar')


args = parser.parse_args()

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

max_seq_len = 80
k = 1

with open('data/hred/Vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)

with open('data/hred/train.pkl', 'rb') as f:
    train_data = pickle.load(f)  # (max_turns, max_seq_len, data_size)

with open('data/hred/valid.pkl', 'rb') as f:
    valid_data = pickle.load(f)

with open('data/hred/test.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(len(itos))

model = HRED(
    emb_dim=300, n_vocab=len(itos), h_dim=args.h_dim,
    emb_drop=args.emb_drop, start_token=1, gpu=args.gpu
)

solver = optim.Adam(model.parameters(), lr=args.lr)


def main():
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        model.train()

        total_loss = 0

        for i in tqdm(range(0, train_data.shape[2], args.mb_size)):
            inputs = train_data[:, :max_seq_len, i:i+args.mb_size]
            inputs = Variable(torch.LongTensor(inputs))
            inputs = inputs.cuda() if args.gpu else inputs

            loss = model(inputs)

            loss.backward()
            clip_gradient_threshold(model, -10, 10)
            solver.step()
            solver.zero_grad()

            # If current iteration is multiple of batchsize or it's the last one
            if i % 1000 == 0:
                print('Loss: {:.3f}'.format(loss.data[0]))

#         # Validation
#         recall_at_ks = eval_model_v1(
#             model, udc, 'valid', gpu=args.gpu, no_tqdm=args.no_tqdm
#         )

#         print('Loss: {:.3f}; recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
#               .format(loss.data[0], recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))

#         save_model(model, 'dualencoder')

#     eval_test()


# def eval_test():
#     print('\n\nEvaluating on test set...')
#     print('-------------------------------')

#     recall_at_ks = eval_model_v1(
#         model, udc, 'test', gpu=args.gpu, no_tqdm=args.no_tqdm
#     )

#     print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
#           .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


try:
    main()
    eval_test()
except KeyboardInterrupt:
    eval_test()
    exit(0)
