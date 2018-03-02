from torchtext import data
from torchtext.vocab import GloVe
import torch
import re


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
    return res.split()


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
    for it in range(n_iter):
        contexts, responses, labels = udc.next_train_batch()
        pred = model(contexts, responses)
        loss = compute_loss(pred, labels)
    ```
    """

    def __init__(self, path='data', train_file='train10k.tsv', valid_file='valid500.tsv', test_file='test500.tsv', batch_size=32, embed_dim=100, max_vocab_size=10000, min_freq=5, gpu=False):
        self.TEXT = data.Field(
            lower=True, tokenize=custom_tokenizer,
            unk_token='__unk__', pad_token='__pad__'
        )
        self.LABEL = data.Field(sequential=False, tensor_type=torch.FloatTensor, unk_token=None)

        self.train, self.valid, self.test = data.TabularDataset.splits(
            path=path, train=train_file, validation=valid_file, test=test_file,
            format='tsv', skip_header=True,
            fields=[('context', self.TEXT), ('response', self.TEXT), ('label', self.LABEL)],
        )

        device = 0 if gpu else -1
        sort_key = lambda x: data.interleave_keys(len(x.context), len(x.response))

        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.valid, self.test), batch_size=batch_size, device=device,
            shuffle=True, sort_key=sort_key
        )

        self.TEXT.build_vocab(
            self.train, max_size=max_vocab_size, min_freq=min_freq,
            vectors=GloVe('6B', dim=embed_dim)
        )
        self.LABEL.build_vocab(self.train)

        self.vocab_size = len(self.TEXT.vocab.itos)
        self.embed_dim = embed_dim
        self.vectors = self.TEXT.vocab.vectors

    def next_train_batch(self):
        return next(iter(self.train_iter))

    def next_valid_batch(self):
        return next(iter(self.valid_iter))

    def next_test_batch(self):
        return next(iter(self.test_iter))
