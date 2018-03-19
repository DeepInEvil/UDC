from torchtext import data
from torchtext.vocab import Vocab, GloVe
import torch
import re
import twokenize
from collections import OrderedDict, Counter
import numpy as np

URL_TOK = '__url__'
PATH_TOK = '__path__'


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def custom_tokenizer(text):
    """
    Preprocess and tokenize a text:
    -------------------------------
    1. Replace urls with '__url__'
    2. Replace system paths with '__path__'
    3. Tokenize by whitespace, i.e. str.split()
    """
    res = text

    # URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', res)
    for url in urls:
        res = res.replace(url, URL_TOK)

    # System paths
    paths = re.findall(r'\/.*\.[\w:]+', res)
    for p in paths:
        res = res.replace(p, PATH_TOK)

    # Tokenize
    # return twokenize.tokenize(res)
    return res.split()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()

class UDC:
    """
    Ubuntu Dialogue Corpus wrapper.

    Example usage:
    --------------
    ```
    udc = UDC()

    # Initialize word embeddings
    word_embed = nn.Embedding(udc.vocab_size, udc.embed_dim)
    word_embed.weight.data.copy_(udc.vectors)

    # Training loop
    for epoch in range(n_epoch):
        for mb in udc.train_iter():
            contexts, responses, labels = mb
            pred = model(contexts, responses)
            loss = compute_loss(pred, labels)
            loss.backward()

        for mb in udc.valid_iter():
            contexts, positives, negatives_1, ..., negatives_9 = mb

            for responses in cat(positives, negatives_1, ... negatives_9)
                pred = model(contexts, responses)
    ```
    """

    def __init__(self, path='data', glove_p='glove', train_file='train.csv', valid_file='valid.csv', test_file='test.csv', vocab_file=None, batch_size=32, embed_dim=100, max_vocab_size=None, min_freq=1, max_seq_len=None, gpu=False, use_fasttext=False):
        self.batch_size = batch_size
        self.device = 0 if gpu else -1
        self.sort_key = lambda x: len(x.context)

        self.TEXT = data.Field(
            lower=True, fix_length=max_seq_len,
            pad_token='__pad__', unk_token='<UNK>', batch_first=True, tokenize=clean_str
        )

        self.LABEL = data.Field(
            sequential=False, tensor_type=torch.FloatTensor, unk_token=None,
            batch_first=True
        )

        file_format = train_file[-3:]

        # Only take data with max length 160
        # f = lambda ex: len(ex.context) <= max_seq_len and len(ex.response)
        f = None

        self.train = data.TabularDataset(
            path='{}/{}'.format(path, train_file), format=file_format, skip_header=True,
            fields=[('context', self.TEXT), ('response', self.TEXT), ('label', self.LABEL)],
            filter_pred=f
        )

        self.valid, self.test = data.TabularDataset.splits(
            path=path, validation=valid_file, test=test_file,
            format=file_format, skip_header=True,
            fields=[('context', self.TEXT), ('positive', self.TEXT),
                    ('negative_1', self.TEXT), ('negative_2', self.TEXT),
                    ('negative_3', self.TEXT), ('negative_4', self.TEXT),
                    ('negative_5', self.TEXT), ('negative_6', self.TEXT),
                    ('negative_7', self.TEXT), ('negative_8', self.TEXT),
                    ('negative_9', self.TEXT)]
        )

        if vocab_file is None:

            if use_fasttext:
                print ("building vocabulary")
                # self.TEXT.build_vocab(
                #     self.train, max_size=max_vocab_size, min_freq=3,
                #     vectors="fasttext.en.300d"
                # )
                self.TEXT.build_vocab(
                    self.train, max_size=max_vocab_size, min_freq=6,
                    vectors = "fasttext.en.300d"
                )
            else:
                self.TEXT.build_vocab(
                    self.train, max_size=max_vocab_size, min_freq=min_freq,
                    vectors=GloVe('6B', dim=embed_dim)
                )
            vocab = self.TEXT.vocab
        else:
            specials = list(OrderedDict.fromkeys(
                tok for tok in [self.TEXT.unk_token, self.TEXT.pad_token,
                                self.TEXT.init_token, self.TEXT.eos_token]
                if tok is not None))

            with open(f'{path}/{vocab_file}', 'r') as f:
                counter = Counter(f.read().split('\n'))

            if use_fasttext:
                print ("Using fasttext")
                vocab = Vocab(counter, specials=specials,
                              vectors="fasttext.en.300d")
            else:
                vocab = Vocab(counter, specials=specials,
                              vectors=GloVe('6B', dim=embed_dim))

            self.TEXT.vocab = vocab

        self.LABEL.build_vocab(self.train)
        print (vocab.stoi['__pad__'])
        print (vocab.itos[0])
        self.dataset_size = len(self.train.examples)
        self.vocab_size = len(self.TEXT.vocab.itos)
        self.embed_dim = embed_dim
        #self.vectors = self.load_glove_embeddings(glove_p+'/glove.6B.50d.txt', self.TEXT.vocab.stoi)
        self.vectors = self.TEXT.vocab.vectors

    def load_glove(self, path):
        """
        creates a dictionary mapping words to vectors from a file in glove format.
        """
        with open(path) as f:
            glove = {}
            for line in f.readlines():
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                glove[word] = vector
            return glove

    def load_glove_embeddings(self, path, word2idx, embedding_dim=50):
        with open(path) as f:
            embeddings = np.zeros((len(word2idx), embedding_dim))
            for line in f.readlines():
                values = line.split()
                word = values[0]
                index = word2idx.get(word)
                if index:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[index] = vector
            return torch.from_numpy(embeddings).float()

    def train_iter(self):
        train_iter = data.BucketIterator(
            self.train, batch_size=self.batch_size, device=self.device,
            shuffle=True, sort_key=self.sort_key, train=True, repeat=False
        )
        return iter(train_iter)

    def valid_iter(self):
        valid_iter = data.BucketIterator(
            self.valid, batch_size=self.batch_size, device=self.device,
            sort_key=self.sort_key, shuffle=False, train=False, repeat=False
        )
        return iter(valid_iter)

    def test_iter(self):
        test_iter = data.BucketIterator(
            self.test, batch_size=self.batch_size, device=self.device,
            sort_key=self.sort_key, shuffle=False, train=False, repeat=False
        )
        return iter(test_iter)
