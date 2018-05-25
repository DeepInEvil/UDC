import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init
from tqdm import tqdm
from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM, EmbMM
from data import UDCv1, UDCv2, UDCv4
from util import save_model, clip_gradient_threshold
from DeepAttention import LSTMDualAttnEnc, LSTMPAttn, GRUDualAttnEnc, GRUAttnmitKey, LSTMKeyAttn, GRUAttn_KeyCNN4
from util import load_model

#load vocab and other functions

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
        return '__PAD__'


def get_words(sent):
    #print (sent)
    sents = [getI2W(w) for w in sent]
    return ' '.join(sents)


def get_atten_dict(sent, weights):
    out_attn = []
    for i, word in enumerate(sent):
        out_dict = {}
        if word != 0:
            out_dict['word'] = getI2W(word)
            out_dict['attention'] = weights[i]
            out_attn.append(out_dict)
    return out_attn


udc = UDCv4('ubuntu_data', batch_size=64, use_mask=True,
            max_seq_len=320, gpu=True, use_fasttext=True)

model = GRUAttn_KeyCNN4(udc.emb_dim, udc.vocab_size, 300, udc.vectors, 0, True)
model = load_model(model, 'GRU_kb_enc_best')
model.eval()

data_iter = udc.get_iter('test')

attentions = []
data_iter = tqdm(data_iter)
data_iter.set_description_str('Testing')
n_data = udc.n_test
data_iter.total = n_data // udc.batch_size

for mb in data_iter:
    context, response, y, cm, rm, _, key_r, key_m_r = mb
    key_mask_r = key_m_r.unsqueeze(2).repeat(1, 1, 50 * 4)
    scores_mb = F.sigmoid(model(context, response, cm, rm, key_r, key_m_r)).cpu().data.numpy()
    #scores_mb = F.sigmoid(model(context, response, cm, rm)).cpu().data.numpy()
    key_emb_r = model.get_weighted_key(key_r, key_mask_r)
    sc, sr, c, r = model.forward_enc(context, response, key_emb_r)

    max_len = sr.size(1)
    b_size = sr.size(0)

    c = c.squeeze(0).unsqueeze(2)
    attn = model.attn(sr.contiguous().view(b_size * max_len, -1))  # B*T,D -> B*T,D
    attn = attn.view(b_size, max_len, -1)  # B,T,D
    attn_energies = (attn.bmm(c).transpose(1, 2))  # B,T,D * B,D,1 --> B,1,T
    alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
    alpha = alpha * rm
    for i, rb in enumerate(response):
        if (torch.sum(key_m_r[i]).cpu().data.numpy()) > 0 and scores_mb[i] > 0.5 and y[i].cpu().data.numpy() == 1:
            attentions.append(get_atten_dict(response[i].cpu().data.numpy(), alpha[i].cpu().data.numpy()))

np.save('ubuntu_data/attention_sigmod.npy', attentions)