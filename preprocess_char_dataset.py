import pickle


with open('/home/DebanjanChaudhuri/UDC/ubuntu_data/dataset_1Mstr.pkl', 'rb') as f:
    dataset = pickle.load(f, encoding='ISO-8859-1')
    train_char, valid_char, test_char, _, _ = dataset

# Get vocabulary of characters
chars = set(['__UNK__'])  # Add special unknown char


def get_unique_chars(dataset):
    global chars

    for sent in dataset:
        for word in sent:
            chars = chars.union(set(word))


for d in (train_char['c'], train_char['r']):
    get_unique_chars(d)

ctoi = dict()
itoc = ['']

# Index 0 is for padding => i+1
for i, c in enumerate(chars):
    ctoi[c] = i+1
    itoc.append(c)

with open('vocab.pkl', 'wb') as f:
    pickle.dump((ctoi, itoc), f)

print(itoc)


def preprocess_set(dataset):
    preped_data = []

    for sent in dataset:
        preped_sent = []
        for word in sent:
            preped_word = []
            for char in word:
                try:
                    idx = ctoi[char]
                except KeyError:
                    idx = ctoi['__UNK__']
                preped_word.append(idx)
            preped_sent.append(preped_word)
        preped_data.append(preped_sent)

    return preped_data


def preprocess(dataset):
    """ Preprocess char into its index """
    preped_dataset = {}
    preped_dataset['c'] = preprocess_set(dataset['c'])
    preped_dataset['r'] = preprocess_set(dataset['r'])
    preped_dataset['y'] = dataset['y']
    return preped_dataset


# Preprocess train, valid, and test set
dataset = [preprocess(d) for d in (train_char, valid_char, test_char)]
dataset.append(ctoi)
dataset.append(itoc)

# Repack dataset
with open('dataset_1Mstr_preped.pkl', 'wb') as f:
    pickle.dump(dataset, f)
