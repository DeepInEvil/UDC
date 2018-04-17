import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init

from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM, EmbMM
from data import UDCv1, UDCv2, UDCv3
from util import save_model, clip_gradient_threshold
from DeepAttention import LSTMDualAttnEnc, LSTMPAttn, GRUDualAttnEnc, GRUAttnmitKey, LSTMKeyAttn, GRUAttn_KeyCNN
from util import load_model

from tqdm import tqdm

udc = UDCv2('/home/DebanjanChaudhuri/UDC/ubuntu_data', batch_size=250, use_mask=True,
            max_seq_len=320, gpu=True, use_fasttext=True)

model = GRUDualAttnEnc(
    udc.emb_dim, udc.vocab_size, 300, udc.vectors, 0, True
)


def eval_test():
    print('\n\nEvaluating on test set...')
    print('-------------------------------')

    recall_at_ks = eval_model_v1(
        model, udc, 'test', gpu=True, no_tqdm=True
    )

    print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


def recall_at_k_np(scores, ks=[1, 2, 3, 4, 5]):
    sorted_idxs = np.argsort(-scores, axis=1)
    ranks = (sorted_idxs == 0).argmax(1)
    recalls = [np.mean(ranks+1 <= k) for k in ks]
    return recalls


def eval_model_v1(model, dataset, mode='valid', gpu=False, no_tqdm=False):
    model.eval()
    scores = []

    assert mode in ['valid', 'test']

    data_iter = dataset.get_iter(mode)

    if not no_tqdm:
        data_iter = tqdm(data_iter)
        data_iter.set_description_str('Evaluation')
        n_data = dataset.n_valid if mode == 'valid' else dataset.n_test
        data_iter.total = n_data // dataset.batch_size

    for mb in data_iter:
        context, response, y, cm, rm = mb

        # Get scores
        scores_mb = F.sigmoid(model(context, response, cm))
        scores_mb = scores_mb.cpu() if gpu else scores_mb
        scores.append(scores_mb.data.numpy())
    scores = np.concatenate(scores)
    print (scores.shape)
    scores = scores[:-(scores.shape[0] % 10)]
    scores = scores.reshape(-1, 10)  # 1 in 10
    recall_at_ks = [r for r in recall_at_k_np(scores)]

    return recall_at_ks


model = load_model(model, 'GRU_key_enc')

eval_test()