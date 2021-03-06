import pandas as pd
import numpy as np
import re
import pickle
from collections import defaultdict
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


def get_values(file, get_c_d=False):
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
    vocab = open('./vocab.txt', 'r').readlines()
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
    train['y'], train['c'], train['r'], char_d = get_values('./ubuntu_data/train.txt', get_c_d=True)
    test['y'], test['c'], test['r'] = get_values('./ubuntu_data/test.txt')
    valid['y'], valid['c'], valid['r'] = get_values('./ubuntu_data/valid.txt')
    char_vocab = defaultdict(float)
    print (char_d)
    dataset = train, valid, test, w2id, char_d
    pickle.dump(dataset, open('dataset_1Mstr.pkl', 'wb'))


def get_queries(sent):



for i, r in enumerate(test['r']):
    resp = r
    label = test['y'][i]
    query = ' '.join(test['c'][i]).split('63346')[-1].split()
    try:
        if int(query[-1]) in q_idx and int(r[0]) in [124405, 124413, 54469] and label==0:
        #if int(query[0]) in [124405, 124413, 54469] and int(query[-1]) in q_idx and int(r[0]) in [124405, 124413, 54469] and label==1:
            print (query, r)
    except IndexError:
        continue

o = 0
for c, r, p in wrongs:
    last_utr = c.split('__EOS__')[-1].strip()

    try:
        if (last_utr.split()[0] in ['what','how','whay','why']) and p.split()[0] in ['what','how','whay','why', 'when']:
            print (last_utr)
            print ("***********")
            print ("responses")
            print (r )
            print ('==========')
            print (p)
            o += 1
        if (last_utr.split()[-1] in q_m) and p.split()[0] in ['what','how','whay','why', 'when']:
            print (last_utr)
            print ("***********")
            print ("responses")
            print (r )
            print ('==========')
            print (p)
            o += 1
        if (last_utr.split()[-1] in q_m) and p.split()[-1] in q_m:
            print (last_utr)
            print ("***********")
            print ("responses")
            print (r )
            print ('==========')
            print (p)
            o += 1
    except IndexError:
        continue



o = 0
for c, r, p in wrongs:
    last_utr = c.split('__EOS__')[-1].strip()

    try:
        if last_utr.split()[0] in ['what','how','whay','why'] and (r.split()[0] in ['what','how','whay','why']):
            print (last_utr)
            print ("***********")
            print ("responses")
            print (r )
            print ('==========')
            print (p)
            o += 1
    except IndexError:
        continue

def get_freq(file):
    with open(file,'r') as f:
        data = f.readlines()
    for d in data:
        d = d.replace('\n', '').split('\t')
        for c in d:
            for w in c.split():
                vocab[w] += 1.0


import math


def Entropy(string, base=2.0):
    # make set with all unrepeatable symbols from string
    dct = dict.fromkeys(list(string))

    # calculate frequencies
    pkvec = [float(string.count(c)) / len(string) for c in dct]
    print (pkvec)
    # calculate Entropy
    H = -sum([pk * math.log(pk) / math.log(base) for pk in pkvec])
    return H


print(Entropy("Python is not so easy"))


def get_entropy(string, base=2.0):
    words = string.split()
    probs = [prob[w] for w in words]
    H = -sum([math.log(pk) / math.log(base) for pk in probs if pk != 0.0])
    ent = H/len(words)
    return ent
