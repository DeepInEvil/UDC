from torchtext import data
from torchtext.vocab import GloVe
import torch
import re
import twokenize


URL_TOK = '__url__'
PATH_TOK = '__path__'


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
    return twokenize.tokenize(res)


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

    def __init__(self, path='data', train_file='train.csv', valid_file='valid.csv', test_file='test.csv', batch_size=32, embed_dim=100, max_vocab_size=10000, min_freq=5, gpu=False):
        self.batch_size = batch_size
        self.device = 0 if gpu else -1
        self.sort_key = lambda x: len(x.context)

        self.TEXT = data.Field(
            lower=True, tokenize=custom_tokenizer,
            unk_token='__unk__', pad_token='__pad__'
        )
        self.LABEL = data.Field(sequential=False, tensor_type=torch.FloatTensor, unk_token=None)

        file_format = train_file[-3:]

        self.train = data.TabularDataset(
            path='{}/{}'.format(path, train_file), format=file_format, skip_header=True,
            fields=[('context', self.TEXT), ('response', self.TEXT), ('label', self.LABEL)],
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

        self.TEXT.build_vocab(
            self.train, max_size=max_vocab_size, min_freq=min_freq,
            vectors=GloVe('6B', dim=embed_dim)
        )
        self.LABEL.build_vocab(self.train)

        self.dataset_size = len(self.train.examples)
        self.vocab_size = len(self.TEXT.vocab.itos)
        self.embed_dim = embed_dim
        self.vectors = self.TEXT.vocab.vectors

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
