from torchtext import data
from torchtext.vocab import Vocab, GloVe
import torch
from torch.autograd import Variable
import re
from collections import OrderedDict, Counter
import numpy as np
import pickle

URL_TOK = '__url__'
PATH_TOK = '__path__'


class UDCv1:
    """
    Wrapper for UDCv1 taken from: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/.
    Everything has been preprocessed and converted to numerical indexes.
    """

    def __init__(self, path, batch_size=256, max_seq_len=160, use_mask=False, gpu=True, use_fasttext=False):
        self.batch_size = batch_size
        self.max_seq_len_c = max_seq_len
        self.max_seq_len_r = int(max_seq_len/2)
        self.use_mask = use_mask
        self.gpu = gpu

        with open(f'{path}/dataset_1M.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.train, self.valid, self.test = dataset

        with open(f'{path}/dataset_1Mstr_preped.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.char_train, self.char_valid, self.char_test, self.ctoi, self.itoc = dataset

        if use_fasttext:
            vectors = np.load(f'{path}/fast_text_200_v.npy')
        else:
            with open(f'{path}/W.pkl', 'rb') as f:
                vectors, _ = pickle.load(f, encoding='ISO-8859-1')

        print('Finished loading dataset!')

        self.n_train = len(self.train['y'])
        self.n_valid = len(self.valid['y'])
        self.n_test = len(self.test['y'])
        self.vectors = torch.from_numpy(vectors.astype(np.float32))

        self.vocab_size = self.vectors.size(0)
        self.emb_dim = self.vectors.size(1)

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
            char_dataset = self.char_train
        elif dataset == 'valid':
            dataset = self.valid
            char_dataset = self.char_valid
        else:
            dataset = self.test
            char_dataset = self.char_test

        for i in range(0, len(dataset['y']), self.batch_size):
            c = dataset['c'][i:i+self.batch_size]
            r = dataset['r'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]

            char_c = char_dataset['c'][i:i+self.batch_size]
            char_r = char_dataset['r'][i:i+self.batch_size]

            c, r, y, c_mask, r_mask = self._load_batch(c, r, y, char_c, char_r, self.batch_size)

            if self.use_mask:
                yield c, r, y, c_mask, r_mask
            else:
                yield c, r, y

    def _load_batch(self, c, r, y, char_c, char_r, size):
        c_arr = np.zeros([size, self.max_seq_len_c], np.int)
        r_arr = np.zeros([size, self.max_seq_len_r], np.int)
        y_arr = np.zeros(size, np.float32)

        c_mask = np.zeros([size, self.max_seq_len_c], np.float32)
        r_mask = np.zeros([size, self.max_seq_len_r], np.float32)

        max_char_c_seq_len = max([len(x) for x in char_c])
        max_char_r_seq_len = max([len(x) for x in char_r])

        char_c_arr = np.zeros([size, max_char_c_seq_len], np.int)
        char_r_arr = np.zeros([size, max_char_r_seq_len], np.int)

        for j, (row_c, row_r, row_y, row_char_c, row_char_r) in enumerate(zip(c, r, y, char_c_arr, char_r_arr)):
            # Truncate
            row_c = row_c[:self.max_seq_len_c]
            row_r = row_r[:self.max_seq_len_r]

            c_arr[j, :len(row_c)] = row_c
            r_arr[j, :len(row_r)] = row_r
            y_arr[j] = float(row_y)

            c_mask[j, :len(row_c)] = 1
            r_mask[j, :len(row_r)] = 1

            char_c_arr[j, :len(row_char_c)] = row_char_c
            char_r_arr[j, :len(row_char_r)] = row_char_r

        # Convert to PyTorch tensor
        c = Variable(torch.from_numpy(c_arr))
        r = Variable(torch.from_numpy(r_arr))
        y = Variable(torch.from_numpy(y_arr))
        c_mask = Variable(torch.from_numpy(c_mask))
        r_mask = Variable(torch.from_numpy(r_mask))
        char_c = Variable(torch.from_numpy(char_c_arr))
        char_r = Variable(torch.from_numpy(char_r_arr))

        # Load to GPU
        if self.gpu:
            c, r, y = c.cuda(), r.cuda(), y.cuda()
            c_mask, r_mask = c_mask.cuda(), r_mask.cuda()
            char_c, char_r = char_c.cuda(), char_r.cuda()

        return c, r, y, (c_mask, r_mask), (char_c, char_r)


class UDCv2:
    """
    Wrapper for UDCv2 taken from: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/.
    Everything has been preprocessed and converted to numerical indexes.
    """

    def __init__(self, path, batch_size=256, max_seq_len=160, use_mask=False, gpu=True, use_fasttext=False):
        self.batch_size = batch_size
        self.max_seq_len_c = max_seq_len
        self.max_seq_len_r = int(max_seq_len/2)
        self.use_mask = use_mask
        self.gpu = gpu

        with open(f'{path}/dataset_1M.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.train, self.valid, self.test = dataset

        if use_fasttext:
            vectors = np.load(f'{path}/fast_text_200_v.npy')
            #vectors = np.load(f'{path}/w2vec_200.npy')
            #man_vec = np.load(f'{path}/key_vec.npy')
        else:
            with open(f'{path}/W.pkl', 'rb') as f:
                vectors, _ = pickle.load(f, encoding='ISO-8859-1')

        print('Finished loading dataset!')

        self.n_train = len(self.train['y'])
        self.n_valid = len(self.valid['y'])
        self.n_test = len(self.test['y'])
        self.vectors = torch.from_numpy(vectors.astype(np.float32))
        #self.man_vec = torch.from_numpy(man_vec.astype(np.float32))

        self.vocab_size = self.vectors.size(0)
        self.emb_dim = self.vectors.size(1)

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['y']), self.batch_size):
            c = dataset['c'][i:i+self.batch_size]
            r = dataset['r'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]


            c, r, y, c_mask, r_mask = self._load_batch(c, r, y, self.batch_size)

            if self.use_mask:
                yield c, r, y, c_mask, r_mask
            else:
                yield c, r, y

    def _load_batch(self, c, r, y, size):
        c_arr = np.zeros([size, self.max_seq_len_c], np.int)
        r_arr = np.zeros([size, self.max_seq_len_r], np.int)
        y_arr = np.zeros(size, np.float32)

        c_mask = np.zeros([size, self.max_seq_len_c], np.float32)
        r_mask = np.zeros([size, self.max_seq_len_r], np.float32)

        for j, (row_c, row_r, row_y) in enumerate(zip(c, r, y)):
            # Truncate
            row_c = row_c[:self.max_seq_len_c]
            row_r = row_r[:self.max_seq_len_r]

            c_arr[j, :len(row_c)] = row_c
            r_arr[j, :len(row_r)] = row_r
            y_arr[j] = float(row_y)

            c_mask[j, :len(row_c)] = 1
            r_mask[j, :len(row_r)] = 1

        # Convert to PyTorch tensor
        c = Variable(torch.from_numpy(c_arr))
        r = Variable(torch.from_numpy(r_arr))
        y = Variable(torch.from_numpy(y_arr))
        c_mask = Variable(torch.from_numpy(c_mask))
        r_mask = Variable(torch.from_numpy(r_mask))

        # Load to GPU
        if self.gpu:
            c, r, y = c.cuda(), r.cuda(), y.cuda()
            c_mask, r_mask = c_mask.cuda(), r_mask.cuda()

        return c, r, y, c_mask, r_mask


class UDCv3:
    """
    Wrapper for UDCv2 taken from: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/.
    Everything has been preprocessed and converted to numerical indexes.
    """

    def __init__(self, path, batch_size=256, max_seq_len=160, use_mask=False, gpu=True, use_fasttext=False):
        self.batch_size = batch_size
        self.max_seq_len_c = max_seq_len
        self.max_seq_len_r = int(max_seq_len/2)
        self.use_mask = use_mask
        self.gpu = gpu

        with open(f'{path}/dataset_1M.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.train, self.valid, self.test = dataset

        if use_fasttext:
            vectors = np.load(f'{path}/fast_text_200_v.npy')
            #vectors = np.load(f'{path}/w2vec_200.npy')
            #man_vec = np.load(f'{path}/key_vec.npy')
        else:
            with open(f'{path}/W.pkl', 'rb') as f:
                vectors, _ = pickle.load(f, encoding='ISO-8859-1')

        print('Finished loading dataset!')

        self.q_idx = list(np.load('ubuntu_data/ques.npy'))
        self.w1h = [124405, 124413, 54469, 17261]

        self.n_train = len(self.train['y'])
        self.n_valid = len(self.valid['y'])
        self.n_test = len(self.test['y'])
        self.vectors = torch.from_numpy(vectors.astype(np.float32))
        #self.man_vec = torch.from_numpy(man_vec.astype(np.float32))

        self.vocab_size = self.vectors.size(0)
        self.emb_dim = self.vectors.size(1)

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['y']), self.batch_size):
            c = dataset['c'][i:i+self.batch_size]
            r = dataset['r'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]


            c, r, y, c_mask, r_mask, ql = self._load_batch(c, r, y, self.batch_size)

            if self.use_mask:
                yield c, r, y, c_mask, r_mask, ql
            else:
                yield c, r, y

    def _load_batch(self, c, r, y, size):
        c_arr = np.zeros([size, self.max_seq_len_c], np.int)
        r_arr = np.zeros([size, self.max_seq_len_r], np.int)
        y_arr = np.zeros(size, np.float32)

        q_l = np.zeros(size, np.float32)


        c_mask = np.zeros([size, self.max_seq_len_c], np.float32)
        r_mask = np.zeros([size, self.max_seq_len_r], np.float32)

        for j, (row_c, row_r, row_y) in enumerate(zip(c, r, y)):
            #check if query
            try:
                idx_eos = row_c.index(63346, -1)
            except ValueError:
                idx_eos = -1
            last_utr = row_c[idx_eos+1:]
            if int(row_c[-1]) in self.q_idx and int(last_utr[0]) in self.w1h:
                if int(row_r[-1]) in self.q_idx or int(row_r[-1]) in self.w1h:
                    q_l[j] = 1
            # if int(last_utr[0]) in self.w1h:
            #     if int(row_r[-1]) in self.q_idx or int(row_r[-1]) in self.w1h:
            #         q_l[j] = 1

            # Truncate
            row_c = row_c[:self.max_seq_len_c]
            row_r = row_r[:self.max_seq_len_r]

            c_arr[j, :len(row_c)] = row_c
            r_arr[j, :len(row_r)] = row_r
            y_arr[j] = float(row_y)


            c_mask[j, :len(row_c)] = 1
            r_mask[j, :len(row_r)] = 1

        # Convert to PyTorch tensor
        c = Variable(torch.from_numpy(c_arr))
        r = Variable(torch.from_numpy(r_arr))
        y = Variable(torch.from_numpy(y_arr))
        c_mask = Variable(torch.from_numpy(c_mask))
        r_mask = Variable(torch.from_numpy(r_mask))
        q_l = Variable(torch.from_numpy(q_l))

        # Load to GPU
        if self.gpu:
            c, r, y = c.cuda(), r.cuda(), y.cuda()
            c_mask, r_mask = c_mask.cuda(), r_mask.cuda()
            q_l = q_l.cuda()

        return c, r, y, c_mask, r_mask, q_l


class UDCv4:
    """
    Wrapper for UDCv2 taken from: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/.
    Everything has been preprocessed and converted to numerical indexes.
    """

    def __init__(self, path, batch_size=256, max_seq_len=160, use_mask=False, gpu=True, use_fasttext=False):
        self.batch_size = batch_size
        self.max_seq_len_c = max_seq_len
        self.max_seq_len_r = int(max_seq_len/2)
        self.use_mask = use_mask
        self.gpu = gpu

        self.desc_len = 44

        with open(f'{path}/dataset_1M.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.train, self.valid, self.test = dataset

        if use_fasttext:
            vectors = np.load(f'{path}/fast_text_200_v.npy')
            #vectors = np.load(f'{path}/w2vec_200.npy')
            #man_vec = np.load(f'{path}/key_vec.npy')
        else:
            with open(f'{path}/W.pkl', 'rb') as f:
                vectors, _ = pickle.load(f, encoding='ISO-8859-1')
        self.ubuntu_cmd_vec = np.load(f'{path}/man_dict_new.npy').item()
        #self.ubuntu_cmd_vec = np.load(f'{path}/man_dict_key.npy').item()

        print('Finished loading dataset!')

        self.q_idx = list(np.load('ubuntu_data/ques.npy'))
        self.w1h = [124405, 124413, 54469, 17261]

        self.n_train = len(self.train[  'y'])
        self.n_valid = len(self.valid['y'])
        self.n_test = len(self.test['y'])
        self.vectors = torch.from_numpy(vectors.astype(np.float32))
        #self.man_vec = torch.from_numpy(man_vec.astype(np.float32))

        self.vocab_size = self.vectors.size(0)
        self.emb_dim = self.vectors.size(1)

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['y']), self.batch_size):
            c = dataset['c'][i:i+self.batch_size]
            r = dataset['r'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]


            c, r, y, c_mask, r_mask, ql, key_r, key_mask_r = self._load_batch(c, r, y, self.batch_size)

            if self.use_mask:
                yield c, r, y, c_mask, r_mask, ql, key_r, key_mask_r
            else:
                yield c, r, y


    def get_key(self, sentence, max_seq_len, max_len):
        """
        get key mask
        :param sentence:
        :param max_len:
        :return:
        """
        key_mask = np.zeros((max_seq_len))
        keys = np.zeros((max_seq_len, max_len))
        for j, word in enumerate(sentence):
            if int(word) in self.ubuntu_cmd_vec.keys():
                keys[j] = self.ubuntu_cmd_vec[int(word)][:max_len]
                key_mask[j] = 1
            else:
                keys[j] = np.zeros((max_len))
        return key_mask, keys


    def _load_batch(self, c, r, y, size):
        c_arr = np.zeros([size, self.max_seq_len_c], np.int)
        r_arr = np.zeros([size, self.max_seq_len_r], np.int)
        y_arr = np.zeros(size, np.float32)

        q_l = np.zeros(size, np.float32)


        c_mask = np.zeros([size, self.max_seq_len_c], np.float32)
        r_mask = np.zeros([size, self.max_seq_len_r], np.float32)

        #key_c = np.zeros([size, self.max_seq_len_c, self.desc_len], np.float32)
        key_r = np.zeros([size, self.max_seq_len_r, self.desc_len], np.float32)

        #key_mask_c = np.zeros([size, self.max_seq_len_c], np.float32)
        key_mask_r = np.zeros([size, self.max_seq_len_r], np.float32)

        for j, (row_c, row_r, row_y) in enumerate(zip(c, r, y)):
            #check if query
            try:
                idx_eos = row_c.index(63346, -1)
            except ValueError:
                idx_eos = -1
            last_utr = row_c[idx_eos+1:]
            if int(row_c[-1]) in self.q_idx and int(last_utr[0]) in self.w1h:
                if int(row_r[-1]) in self.q_idx or int(row_r[-1]) in self.w1h:
                    q_l[j] = 1
            # if int(last_utr[0]) in self.w1h:
            #     if int(row_r[-1]) in self.q_idx or int(row_r[-1]) in self.w1h:
            #         q_l[j] = 1



            # Truncate
            row_c = row_c[:self.max_seq_len_c]
            row_r = row_r[:self.max_seq_len_r]

            c_arr[j, :len(row_c)] = row_c
            r_arr[j, :len(row_r)] = row_r
            y_arr[j] = float(row_y)


            c_mask[j, :len(row_c)] = 1
            r_mask[j, :len(row_r)] = 1

            #key_mask_c[j], key_c[j] = self.get_key(row_c, self.max_seq_len_c, self.desc_len)
            key_mask_r[j], key_r[j] = self.get_key(row_r, self.max_seq_len_r, self.desc_len)

        # Convert to PyTorch tensor
        c = Variable(torch.from_numpy(c_arr))
        r = Variable(torch.from_numpy(r_arr))
        y = Variable(torch.from_numpy(y_arr))
        c_mask = Variable(torch.from_numpy(c_mask))
        r_mask = Variable(torch.from_numpy(r_mask))
        q_l = Variable(torch.from_numpy(q_l))

        #key_mask_c = Variable(torch.from_numpy(key_mask_c), requires_grad = False)
        key_mask_r = Variable(torch.from_numpy(key_mask_r), requires_grad = False)

        #key_c = Variable(torch.from_numpy(key_c)).type(torch.LongTensor)
        key_r = Variable(torch.from_numpy(key_r)).type(torch.LongTensor)



        # Load to GPU
        if self.gpu:
            c, r, y = c.cuda(), r.cuda(), y.cuda()
            c_mask, r_mask = c_mask.cuda(), r_mask.cuda()
            #r_mask = r_mask.cuda()
            q_l = q_l.cuda()
            #key_c, key_mask_c, key_r, key_mask_r = key_c.cuda(), key_mask_c.cuda(), key_r.cuda(), key_mask_r.cuda()
            key_r, key_mask_r = key_r.cuda(), key_mask_r.cuda()

        return c, r, y, c_mask, r_mask, q_l, key_r, key_mask_r
