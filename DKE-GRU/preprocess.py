import re
import pickle
from collections import defaultdict


def get_values(file, get_c_d=False):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    chars = []
    y = [int(a[0]) for a in data]
    c = [' __EOS__ '.join(a[1:-1]).split() for a in data]
    r = [a[-1].split() for a in data]
    if get_c_d:
        for word in c:
            sent = ' '.join(word)
            for char in sent:
                chars.append(char)
        chars = set(chars)
        return y, c, r, dict(zip(chars, range(len(chars))))
    else:
        return y, c, r


if __name__ == '__main__':
    #load the vocab file
    vocab = open('ubuntu_data/vocab.txt', 'r').readlines()
    w2id = {}
    for word in vocab:
        w = word.split('\n')[0].split('\t')
        w2id[w[0]] =int(w[1])
    #
    # man_cmd = pd.read_csv('./data/man.csv', sep='\t', header=None)
    # man_cmd = np.array(man_cmd)
    #
    # man_dict = {}
    # for a, b in man_cmd:
    #     man_dict[w2id[a.strip()]] = get_vec(b)

    train, test, valid = {}, {}, {}
    train['y'], train['c'], train['r'] = get_values('ubuntu_data/train.txt', get_c_d=False)
    test['y'], test['c'], test['r'] = get_values('ubuntu_data/test.txt')
    valid['y'], valid['c'], valid['r'] = get_values('ubuntu_data/valid.txt')
    char_vocab = defaultdict(float)
    dataset = train, valid, test
    pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))