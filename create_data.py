import pandas as pd
import numpy as np
import re
import pickle
#read train data

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


def get_values(file):
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    y = [int(a[0]) for a in data]
    c = [' __EOS__ '.join(a[1:-1]).split() for a in data]
    r = [a[-1].split() for a in data]
    return y, c, r


def get_vec(sent):
    vec = 0.0
    len = 0.0
    print (sent)
    sent = clean_str(sent)
    for word in sent:
        try:
            vec = vec + fast_text_vec[w2id[word]]
            len = len + 1.0
        except KeyError:
            continue
    return vec/len

if __name__ == '__main__':
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
    train['y'], train['c'], train['r'] = get_values('./ubuntu_data/train.txt')
    test['y'], test['c'], test['r'] = get_values('./ubuntu_data/test.txt')
    valid['y'], valid['c'], valid['r'] = get_values('./ubuntu_data/valid.txt')
    print (train['c'][0])
    dataset = train, valid, test, w2id

    pickle.dump((dataset, open('dataset_1M_string.pkl', 'wb')))


