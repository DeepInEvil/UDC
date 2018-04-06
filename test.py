from data import UDC, UDCv1
from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM
from util import load_model
from DeepAttention import GRUDualAttnEnc

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


max_seq_len = 220
k = 1
h_dim = args.h_dim

udc = UDCv1('data/dataset_1MM', batch_size=args.mb_size, use_mask = True,
            max_seq_len=max_seq_len, gpu=args.gpu)

# model = CNNDualEncoder(dataset.embed_dim, dataset.vocab_size, h_dim, dataset.vectors, args.gpu)
model = GRUDualAttnEnc(
    udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
)
# model = CCN_LSTM(dataset.embed_dim, dataset.vocab_size, h_dim, max_seq_len, k, dataset.vectors, args.gpu)

model = load_model(model, args.model_name, gpu=args.gpu)

model.eval()
scores = []

valid_iter = tqdm(udc.get_iter('test'))
valid_iter.set_description_str('test')

for mb in valid_iter:
    context, response, y, cm, rm = mb

    # Get scores
    scores_mb = F.sigmoid(model(context, response, cm))
    scores_mb = scores_mb.cpu() if args.gpu else scores_mb
    scores.append(scores_mb.data.numpy())

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
