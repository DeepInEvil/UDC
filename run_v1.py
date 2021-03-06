import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from model import CNNDualEncoder, LSTMDualEncoder, CCN_LSTM, EmbMM
from data import UDCv1, UDCv2, UDCv3, UDCv4
from evaluation import eval_model_v4, eval_model_v2, eval_model_v3
from util import save_model, clip_gradient_threshold, load_model
from DeepAttention import LSTMDualAttnEnc, LSTMPAttn, GRUDualAttnEnc, GRUAttnmitKey, LSTMKeyAttn, GRUAttn_KeyCNN2, GRUAttn_KeyCNN4
from model import GRUDualEncoder
import random
import argparse
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
parser.add_argument('--mb_size', type=int, default=128, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
# parser.add_argument('--randseed', type=int, default=666, metavar='',
#                     help='random seed (default: 666)')
parser.add_argument('--no_tqdm', default=False, action='store_true',
                    help='disable tqdm progress bar')


args = parser.parse_args()
rand_int = random.randint(0, 1000)
# Set random seed
#np.random.seed(args.randseed)
np.random.seed(rand_int)
#torch.manual_seed(args.randseed)
torch.manual_seed(rand_int)
print ("Running with random seed:" + str(rand_int))
if args.gpu:
    #torch.cuda.manual_seed(args.randseed)
    torch.cuda.manual_seed(rand_int)

max_seq_len = 320

udc = UDCv4('ubuntu_data', batch_size=args.mb_size, use_mask=True,
            max_seq_len=max_seq_len, gpu=args.gpu, use_fasttext=True)

model = GRUAttn_KeyCNN4(
    udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
)
model_name = 'dke_gru'+str(rand_int)
# model = LSTMPAttn(
#     udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
# )
# model = CCN_LSTM(
#     udc.emb_dim, udc.vocab_size, args.h_dim, max_seq_len, k,
#     udc.vectors, args.gpu, args.emb_drop
# )
#model = CNNDualEncoder(udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, args.gpu, args.emb_drop)

solver = optim.Adam(model.parameters(), lr=args.lr)

if args.gpu:
    model.cuda()


def compute_qloss(ql, y):

    qloss = Variable(torch.zeros(y.size(0))).cuda()
    tot = 0.0
    for i in range(ql.size(0)):
        qloss[i] = ql[i] * F.relu(y[i]) * 0.001
        tot += 1
    return torch.sum(qloss)/tot


def main():
    best_val = 0.0
    for epoch in range(args.n_epoch):
        print('\n\n-------------------------------------------')
        print('Epoch-{}'.format(epoch))
        print('-------------------------------------------')

        model.train()

        train_iter = enumerate(udc.get_iter('train'))

        if not args.no_tqdm:
            train_iter = tqdm(train_iter)
            train_iter.set_description_str('Training')
            train_iter.total = udc.n_train // udc.batch_size

        for it, mb in train_iter:
            #context, response, y, cm, rm, ql = mb
            context, response, y, cm, rm, ql, key_r, key_mask_r = mb
            output = model(context, response, cm, rm, key_r, key_mask_r)
            #output = model(context, response, cm, rm)
            #output = model(context, response)
            loss = F.binary_cross_entropy_with_logits(output, y)
            # loss = F.mse_loss(F.sigmoid(output), y)

            loss.backward()
            #print (model.conv3.grad)
            #clip_gradient_threshold(model, -10, 10)
            solver.step()
            solver.zero_grad()

        del (context, response, y, output)
        # Validation
        recall_at_ks = eval_model_v2(
            model, udc, 'valid', gpu=args.gpu, no_tqdm=args.no_tqdm
        )

        print('Loss: {:.3f}; recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
              .format(loss.data[0], recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))
        recall_1 = recall_at_ks[0]
        # if epoch > 10:
        #     eval_test()

        if best_val == 0.0:
            save_model(model, model_name)
            best_val = recall_1
        else:
            if recall_1 > best_val:
                best_val = recall_1
                print ("Saving model for recall@1:" + str(recall_1))
                save_model(model, model_name)
            else:
                print ("Not saving, best accuracy so far:" + str(best_val))


def eval_test():
    print('\n\nEvaluating on test set...')
    print('-------------------------------')
    print('Loading the best model........')
    model = GRUAttn_KeyCNN4(
        udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
    )    #
    # model = GRUDualEncoder(
    #      udc.emb_dim, udc.vocab_size, args.h_dim, udc.vectors, 0, args.gpu
    # )
    model = load_model(model, model_name)
    model.eval()
    recall_at_ks = eval_model_v2(
        model, udc, 'test', gpu=args.gpu, no_tqdm=args.no_tqdm
    )

    print('Recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
          .format(recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))


try:
    main()
    eval_test()
except KeyboardInterrupt:
    eval_test()
    exit(0)
