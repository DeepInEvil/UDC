from data import UDC
from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM
from util import load_model

import torch
import torch.nn.functional as F

import argparse
from tqdm import tqdm

import pandas as pd


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
parser.add_argument('--max_context_len', type=int, default=160, metavar='',
                    help='max sequence length for context (default: 160)')
parser.add_argument('--max_response_len', type=int, default=80, metavar='',
                    help='max sequence length for response (default: 80)')
parser.add_argument('--toy_data', default=False, action='store_true',
                    help='whether to use toy dataset (10k instead of 1m)')
parser.add_argument('--model_name', metavar='',
                    help='name of model file to be loaded e.g. `ccn_lstm` corresponds to `models/ccn_lstm.bin`')

args = parser.parse_args()


max_seq_len = 160
k = 1
h_dim = args.h_dim

if args.toy_data:
    dataset = UDC(
        train_file='train10k.csv', valid_file='valid500.csv', test_file='test500.csv', vocab_file='vocabulary.txt',
        embed_dim=args.emb_dim, batch_size=args.mb_size, max_seq_len=max_seq_len, gpu=args.gpu
    )
else:
    dataset = UDC(
        train_file='train.csv', valid_file='valid.csv', test_file='test.csv', vocab_file='vocabulary.txt',
        embed_dim=args.emb_dim, batch_size=args.mb_size, max_seq_len=max_seq_len, gpu=args.gpu
    )

# model = CNNDualEncoder(dataset.embed_dim, dataset.vocab_size, h_dim, dataset.vectors, args.gpu)
model = LSTMDualEncoder(dataset.embed_dim, dataset.vocab_size, h_dim, dataset.vectors, args.gpu)
# model = CCN_LSTM(dataset.embed_dim, dataset.vocab_size, h_dim, max_seq_len, k, dataset.vectors, args.gpu)

model = load_model(model, args.model_name, gpu=args.gpu)

model.eval()
scores = []

valid_iter = tqdm(dataset.valid_iter())
valid_iter.set_description_str('Validation')

for mb in tqdm(valid_iter):
    context = mb.context[:, :args.max_context_len]

    # Get score for positive/ground-truth response
    score_pos = F.sigmoid(model(context, mb.positive[:, :args.max_response_len]).unsqueeze(1))
    # Get scores for negative samples
    score_negs = [
        F.sigmoid(model(context, getattr(mb, 'negative_{}'.format(i))[:, :args.max_response_len]).unsqueeze(1))
        for i in range(1, 10)
    ]
    # Total scores, positives at position zero
    scores_mb = torch.cat([score_pos, *score_negs], dim=1)

    scores.append(scores_mb)

scores = torch.cat(scores, dim=0)

print(scores.size())

_, sorted_idxs = torch.sort(scores, dim=1, descending=True)
_, ranks = (sorted_idxs == 0).max(1)

failures = []

for i, r in enumerate(ranks):
    # Rank > 5 => failure case
    if r.data[0] > 5:
        d = dict(
            context=' '.join(dataset.valid.examples[i].context),
            response=' '.join(dataset.valid.examples[i].positive)
        )

        failures.append(d)

print(len(failures))

df = pd.DataFrame(failures)
df.to_csv('failure_cases.csv', index=False)
