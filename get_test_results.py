import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init

from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM, EmbMM
from data import UDCv1, UDCv2, UDCv4
from util import save_model, clip_gradient_threshold
from DeepAttention import LSTMDualAttnEnc, LSTMPAttn, GRUDualAttnEnc, GRUAttnmitKey, LSTMKeyAttn, GRUAttn_KeyCNN4
from util import load_model

from tqdm import tqdm

udc = UDCv4('ubuntu_data', batch_size=10, use_mask=True,
            max_seq_len=320, gpu=True, use_fasttext=True)

model = GRUAttn_KeyCNN4(
    udc.emb_dim, udc.vocab_size, 300, udc.vectors, 0, True
)

vocab = open('ubuntu_data/vocab.txt', 'r').readlines()
w2id = {}
for word in vocab:
    w = word.split('\n')[0].split('\t')
    w2id[w[0]] = int(w[1])

i2w = {v: k for k, v in w2id.items()}


def getI2W(word):
    try:
        return i2w[word]
    except KeyError:
        return ''


def get_words(sent):
    #print (sent)
    sents = [getI2W(w) for w in sent]
    return ' '.join(sents)


def eval_test():
    print('\n\nEvaluating on test set...')
    print('-------------------------------')

    recall_at_ks = eval_model_v1(
        model, udc, 'test', gpu=True, no_tqdm=False
    )

    print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


def recall_at_k_np(scores, ks=[1, 2, 3, 4, 5]):
    sorted_idxs = np.argsort(-scores, axis=1)
    ranks = (sorted_idxs == 0).argmax(1)
    recalls = [np.mean(ranks+1 <= k) for k in ks]
    return recalls


def evaluate_recall(y_pred, k=1):
    """
    Get a batch of scores and compute the recalls
    :param y_pred: predicted batch of candidates batch_size x 10
    :param k: recall test value
    :return:
    """
    _, sorted_idxs = torch.sort(y_pred, dim=0, descending=True)
    num_correct = 0.0
    predictions = sorted_idxs
    if 0 in predictions[:k].cpu().data:
        return True, _
    else:
        return False, predictions[0].cuda()


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

    out_file = open('ubuntu_data/wrong_pred_dke.txt', 'w')
    out_file2 = open('ubuntu_data/correct_pred_dke.txt', 'w')

    for mb in data_iter:
        context, response, y, cm, rm, ql, key_r, key_mask_r = mb

        # Get scores
        scores_mb = F.sigmoid(model(context, response, cm, rm, key_r, key_mask_r))
        scores_mb = scores_mb.cpu() if gpu else scores_mb
        pred = np.argmax(scores_mb.cpu().data.numpy())

        if pred != 0:
            cntxt = get_words(context[0].cpu().data.numpy()).strip()

            correct_response = get_words((response[0].cpu().data.numpy()))
            # print (response[j+_].cpu().data.numpy())
            predicted = get_words((response[pred].cpu().data.numpy()))

            out_file.write(cntxt + '\t' + correct_response + '\t' + predicted)
            out_file.write('\n')
        else:
            cntxt = get_words(context[0].cpu().data.numpy()).strip()
            correct_response = get_words((response[0].cpu().data.numpy()))
            # print (response[j+_].cpu().data.numpy())
            predicted = get_words((response[pred].cpu().data.numpy()))

            out_file2.write(cntxt + '\t' + correct_response + '\t' + predicted)
            out_file2.write('\n')
        scores.append(scores_mb.data.numpy())
    out_file.close()
    out_file2.close()
    scores = np.concatenate(scores)
    mod = scores.shape[0] % 10
    scores = scores[:-mod if mod != 0 else None]

    scores = scores.reshape(-1, 10)  # 1 in 10
    recall_at_ks = [r for r in recall_at_k_np(scores)]

    return recall_at_ks


model = load_model(model, 'DKE-GRU')

eval_test()