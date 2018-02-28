import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn
import datetime
import operator

np.random.seed(0)


def create_dataframe(csvfile):
    dataframe = pd.read_csv(csvfile)
    return dataframe


def shuffle_dataframe(dataframe):
    dataframe.reindex(np.random.permutation(dataframe.index))


def create_vocab(dataframe):
    vocab = []
    word_freq = {}

    for index, row in dataframe.iterrows():

        context_cell = row["Context"]
        response_cell = row["Utterance"]

        train_words = str(context_cell).split() + str(response_cell).split()

        for word in train_words:

            if word.lower() not in vocab:
                vocab.append(word.lower())

            if word.lower() not in word_freq:
                word_freq[word.lower()] = 1
            else:
                word_freq[word] += 1

    word_freq_sorted = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    vocab = ["<UNK>"] + [pair[0] for pair in word_freq_sorted]

    return vocab


def create_word_to_id(vocab):
    enumerate_list = [(id, word) for id, word in enumerate(vocab)]

    word_to_id = {pair[1]: pair[0] for pair in enumerate_list}

    return word_to_id


def create_id_to_vec(word_to_id, glovefile):
    lines = open(glovefile, 'r').readlines()
    id_to_vec = {}
    vector = None

    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype='float32')  # 32

        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))

    for word, id in word_to_id.items():
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape) * 0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))

    embedding_dim = id_to_vec[0].shape[0]

    return id_to_vec, embedding_dim


def load_ids_and_labels(row, word_to_id):
    context_ids = []
    response_ids = []

    context_cell = row['Context']
    response_cell = row['Utterance']
    label_cell = row['Label']

    max_context_len = 160

    context_words = context_cell.split()
    if len(context_words) > max_context_len:
        context_words = context_words[:max_context_len]
    for word in context_words:
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else:
            context_ids.append(0)  # UNK

    response_words = response_cell.split()
    for word in response_words:
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else:
            response_ids.append(0)

    label = np.array(label_cell).astype(np.float32)

    return context_ids, response_ids, label


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 num_directions,
                 dropout,
                 bidirectional,
                 rnn_type,
                 p_dropout):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.bidirectional = False
        self.rnn_type = 'lstm'
        self.p_dropout = p_dropout

        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout=dropout,
                            bidirectional=False)
        self.dropout_layer = nn.Dropout(self.p_dropout)

        self.init_weights()

    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a=-0.01, b=0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True

        embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)

        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        outputs, hiddens_tuple = self.lstm(embeddings)

        last_hidden = hiddens_tuple[
            0]  # access first tuple element to get last hidden state, dimensions: (num_layers * num_directions x batch_size x hidden_size)
        last_hidden = last_hidden[-1]  # access last lstm layer, dimensions: (batch_size x hidden_size)
        last_hidden = self.dropout_layer(last_hidden)  # dimensions: (batch_size x hidden_size)

        return last_hidden


class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad=True)

    def forward(self, context_tensor, response_tensor):
        context_last_hidden = self.encoder(context_tensor)  # dimensions: (batch_size x hidden_size)
        response_last_hidden = self.encoder(response_tensor)  # dimensions: (batch_size x hidden_size)

        # context = context_last_hidden.mm(self.M).cuda()
        context = context_last_hidden.mm(self.M)  # dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size)  # dimensions: (batch_size x 1 x hidden_size)

        response = response_last_hidden.view(-1, self.hidden_size, 1)  # dimensions: (batch_size x hidden_size x 1)

        # score = torch.bmm(context, response).view(-1, 1).cuda()
        score = torch.bmm(context, response).view(-1,
                                                  1)  # dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        return score


def creating_variables(num_training_examples, num_validation_examples, embedding_dim):
    print(str(datetime.datetime.now()).split('.')[0], "Creating variables for training and validation...")

    training_dataframe = create_dataframe('data/training_%d.csv' % num_training_examples)
    vocab = create_vocab(training_dataframe)
    word_to_id = create_word_to_id(vocab)
    id_to_vec, emb_dim = create_id_to_vec(word_to_id, 'glove/glove.6B.%dd.txt' % embedding_dim)

    validation_dataframe = create_dataframe('data/validation_%d.csv' % num_validation_examples)

    print(str(datetime.datetime.now()).split('.')[0], "Variables created.\n")

    return training_dataframe, vocab, word_to_id, id_to_vec, emb_dim, validation_dataframe


def creating_model(hidden_size, p_dropout):
    print(str(datetime.datetime.now()).split('.')[0], "Calling model...")

    encoder = Encoder(
        input_size=emb_dim,
        hidden_size=hidden_size,
        vocab_size=len(vocab),
        num_layers=1,
        dropout=0.0,
        num_directions=1,
        bidirectional=False,
        rnn_type='lstm',
        p_dropout=p_dropout)

    dual_encoder = DualEncoder(encoder)

    print(str(datetime.datetime.now()).split('.')[0], "Model created.\n")
    print(dual_encoder)

    return encoder, dual_encoder


def increase_count(correct_count, score, label):
    if ((score.data[0][0] >= 0.5) and (label.data[0][0] == 1.0)) or (
        (score.data[0][0] < 0.5) and (label.data[0][0] == 0.0)):
        correct_count += 1

    return correct_count


def get_accuracy(correct_count, dataframe):
    accuracy = correct_count / (len(dataframe))

    return accuracy


def train_model(learning_rate, l2_penalty, epochs):
    print(str(datetime.datetime.now()).split('.')[0], "Starting training and validation...\n")
    print("====================Data and Hyperparameter Overview====================\n")
    print("Number of training examples: %d, Number of validation examples: %d" % (
    len(training_dataframe), len(validation_dataframe)))
    print("Learning rate: %.5f, Embedding Dimension: %d, Hidden Size: %d, Dropout: %.2f, L2:%.10f\n" % (
    learning_rate, emb_dim, encoder.hidden_size, encoder.p_dropout, l2_penalty))
    print("================================Results...==============================\n")

    optimizer = torch.optim.Adam(dual_encoder.parameters(), lr=learning_rate, weight_decay=l2_penalty)

    loss_func = torch.nn.BCEWithLogitsLoss()
    # loss_func.cuda()

    best_validation_accuracy = 0.0

    for epoch in range(epochs):

        shuffle_dataframe(training_dataframe)

        sum_loss_training = 0.0

        training_correct_count = 0

        dual_encoder.train()

        for index, row in training_dataframe.iterrows():
            context_ids, response_ids, label = load_ids_and_labels(row, word_to_id)

            context = autograd.Variable(torch.LongTensor(context_ids).view(-1, 1), requires_grad=False)  # .cuda()

            response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1), requires_grad=False)  # .cuda()

            label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1, 1))),
                                      requires_grad=False)  # .cuda()

            score = dual_encoder(context, response)

            loss = loss_func(score, label)

            sum_loss_training += loss.data[0]

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            training_correct_count = increase_count(training_correct_count, score, label)

        training_accuracy = get_accuracy(training_correct_count, training_dataframe)

        # plt.plot(epoch, training_accuracy)

        shuffle_dataframe(validation_dataframe)

        validation_correct_count = 0

        sum_loss_validation = 0.0

        dual_encoder.eval()

        for index, row in validation_dataframe.iterrows():
            context_ids, response_ids, label = load_ids_and_labels(row, word_to_id)

            context = autograd.Variable(torch.LongTensor(context_ids).view(-1, 1))  # .cuda()

            response = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1))  # .cuda()

            label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1, 1))))  # .cuda()

            score = dual_encoder(context, response)

            loss = loss_func(score, label)

            sum_loss_validation += loss.data[0]

            validation_correct_count = increase_count(validation_correct_count, score, label)

        validation_accuracy = get_accuracy(validation_correct_count, validation_dataframe)

        print(str(datetime.datetime.now()).split('.')[0],
              "Epoch: %d/%d" % (epoch, epochs),
              "TrainLoss: %.3f" % (sum_loss_training / len(training_dataframe)),
              "TrainAccuracy: %.3f" % (training_accuracy),
              "ValLoss: %.3f" % (sum_loss_validation / len(validation_dataframe)),
              "ValAccuracy: %.3f" % (validation_accuracy))

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(dual_encoder.state_dict(), 'saved_model_%d_examples.pt' % (len(training_dataframe)))
            print("New best found and saved.")

    print(str(datetime.datetime.now()).split('.')[0], "Training and validation epochs finished.")

if __name__ == '__main__':
    training_dataframe, vocab, word_to_id, id_to_vec, emb_dim, validation_dataframe = creating_variables(
        num_training_examples=1000,
        embedding_dim=50,
        num_validation_examples=100)

    encoder, dual_encoder = creating_model(hidden_size=50,
                                           p_dropout=0.85)

    # encoder.cuda()
    # dual_encoder.cuda

    for name, param in dual_encoder.named_parameters():
        if param.requires_grad:
            print(name)

    train_model(learning_rate=0.001,
                l2_penalty=0.01,
                epochs=100)

    dual_encoder.load_state_dict(torch.load('saved_model_10000_examples.pt'))

    dual_encoder.eval()