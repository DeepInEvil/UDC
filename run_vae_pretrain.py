import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from hybrid_model import GRUDualEncoderPlusVAE
from data import UDCv2
from evaluation import recall_at_k, eval_model_hybrid_v1
from util import save_model

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--emb_dim', type=int, default=100, metavar='',
                    help='embedding dimension (default: 100)')
parser.add_argument('--h_dim', type=int, default=100, metavar='',
                    help='hidden dimension (default: 100)')
parser.add_argument('--z_dim', type=int, default=100, metavar='',
                    help='latent dimension (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--mb_size', type=int, default=128, metavar='',
                    help='size of minibatch (default: 128)')
parser.add_argument('--n_epoch', type=int, default=500, metavar='',
                    help='number of iterations (default: 500)')
parser.add_argument('--randseed', type=int, default=123, metavar='',
                    help='random seed (default: 123)')

args = parser.parse_args()

max_seq_len = 160

# Set random seed
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

if args.gpu:
    torch.cuda.manual_seed(args.randseed)

udc = UDCv2('/home/DebanjanChaudhuri/UDC/ubuntu_data', batch_size=args.mb_size, use_mask=True,
            max_seq_len=max_seq_len, gpu=args.gpu, use_fasttext=True)

eos_idx = 63346
pad_idx = 0
model = GRUDualEncoderPlusVAE(udc.emb_dim, udc.vocab_size, args.h_dim, args.z_dim, udc.vectors, eos_idx, pad_idx, 0.5, args.gpu)

solver = optim.Adam(model.parameters(), lr=1e-3)

n_iter = args.n_epoch * (udc.n_train // udc.batch_size)
kld_start_inc = 3000
kld_weight = 0.01
kld_max = 0.15
kld_inc = (kld_max - kld_weight) / (n_iter - kld_start_inc)

# Pretrain VAE
for epoch in range(args.n_epoch):
    print('\n\n-------------------------------------------')
    print('Epoch-{}'.format(epoch))
    print('-------------------------------------------')

    train_iter = enumerate(udc.get_iter('train'))
    train_iter = tqdm(train_iter)
    train_iter.set_description_str('Training VAE')
    train_iter.total = udc.n_train // udc.batch_size

    for it, mb in train_iter:
        context, response, y, _, _ = mb

        recon_loss, kl_loss = model.forward_vae(context)
        loss = recon_loss + kld_weight*kl_loss

        # Anneal kl_weight
        if it > kld_start_inc and kld_weight < kld_max:
            kld_weight += kld_inc

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 5)
        solver.step()
        solver.zero_grad()

        if it != 0 and it % 1000 == 0:
            print(f'Iter-{it}; recon_loss: {recon_loss.data[0]:.3f}; kl_loss: {kl_loss.data[0]:.3f}')


# for epoch in range(args.n_epoch):
#     print('\n\n-------------------------------------------')
#     print('Epoch-{}'.format(epoch))
#     print('-------------------------------------------')

#     train_iter = enumerate(udc.get_iter('train'))
#     train_iter = tqdm(train_iter)
#     train_iter.set_description_str('Training')
#     train_iter.total = udc.n_train // udc.batch_size

#     for it, mb in train_iter:
#         context, response, y, _, _ = mb

#         outputs_retrieval, recon_loss_c, kl_loss_c, recon_loss_r, kl_loss_r = model(context, response)

#         loss_ret = F.binary_cross_entropy_with_logits(outputs_retrieval, y)
#         loss_vae_c = recon_loss_c + kl_loss_c
#         loss_vae_r = recon_loss_r + kl_loss_r

#         loss = loss_ret + loss_vae_c + loss_vae_r

#         loss.backward()
#         grad_norm = nn.utils.clip_grad_norm(model.parameters(), 10)
#         solver.step()
#         solver.zero_grad()

#         if it != 0 and it % 2000 == 0:
#             # Validation
#             recall_at_ks = eval_model_hybrid_v1(model, udc, 'valid', gpu=args.gpu)

#             print('\nLoss: {:.3f}; recall@1: {:.3f}; recall@2: {:.3f}; recall@5: {:.3f}'
#                 .format(loss.data[0], recall_at_ks[0], recall_at_ks[1], recall_at_ks[4]))

    # save_model(model, 'hybrid_variational')
