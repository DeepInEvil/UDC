import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn.init as init

from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM, EmbMM
from data import UDCv1, UDCv2, UDCv3
from evaluation import eval_model_v1
from util import save_model, clip_gradient_threshold
from DeepAttention import LSTMDualAttnEnc, LSTMPAttn, GRUDualAttnEnc, GRUAttnmitKey, LSTMKeyAttn, GRUAttn_KeyCNN
from util import load_model

udc = UDCv3('/home/DebanjanChaudhuri/UDC/ubuntu_data', batch_size=args.mb_size, use_mask=True,
            max_seq_len=320, gpu=args.gpu, use_fasttext=True)

model = GRUDualAttnEnc(
    udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
)


def eval_test():
    print('\n\nEvaluating on test set...')
    print('-------------------------------')

    recall_at_ks = eval_model_v1(
        model, udc, 'test', gpu=args.gpu, no_tqdm=args.no_tqdm
    )

    print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


model = load_model(model, 'GRU_key_enc')

eval_test()