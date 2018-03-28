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

max_seq_len = 160
k = 1

with open('data/hred/Vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)

with open('data/hred/Training.dialogues.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/hred/Validation.dialogues.pkl', 'rb') as f:
    valid_data = pickle.load(f)

with open('data/hred/Test.dialogues.pkl', 'rb') as f:
    test_data = pickle.load(f)

model = HRED(
    emb_dim=300, n_vocab=len(itos), h_dim=args.h_dim,
    emb_drop=args.emb_drop, max_seq_len=30, start_token=1, gpu=args.gpu
)

solver = optim.Adam(model.parameters(), lr=args.lr)


def chunk_dialogue(dialogue):
    turns = []
    curr_turn = []

    for i, w in enumerate(dialogue):
        curr_turn.append(w)

        if w == 1 or i == len(dialogue)-1:
            if w != 1:  # In the case when last word in dialogue is not <eot>
                curr_turn.append(1)
            curr_turn = Variable(torch.LongTensor(curr_turn))
            curr_turn = curr_turn.cuda() if args.gpu else curr_turn
            turns.append(curr_turn)
            curr_turn = []

    return turns


def main():
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        model.train()

        total_loss = 0

        for i in tqdm(range(len(train_data))):
            # List of LongTensor of len: n_turn. Each entry has len: turn_len
            inputs = chunk_dialogue(train_data[i])
            loss = model(inputs)
            total_loss += loss

            # If current iteration is multiple of batchsize or it's the last one
            if (i > 0 and i % args.mb_size == 0) or i == len(train_data)-1:
                total_loss = total_loss / args.mb_size
                total_loss.backward()
                clip_gradient_threshold(model, -10, 10)
                solver.step()
                solver.zero_grad()

                print('Loss: {:.3f}'.format(total_loss.data[0]))

                total_loss = 0

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