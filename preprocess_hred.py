"""
Output: tensor of (max_turns, max_seq_len, data_size)
"""
import numpy as np
import pickle
import argparse
import tqdm


parser = argparse.ArgumentParser(
    description='UDC Experiment Runner'
)

parser.add_argument('--max_seq_len', type=int, default=160, metavar='',
                    help='max utterance length (default: 160)')
parser.add_argument('--max_turns', type=int, default=3, metavar='',
                    help='max number of turns in one dialogue (default: 3)')

args = parser.parse_args()


with open('data/hred/Vocab.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)

with open('data/hred/Training.dialogues.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/hred/Validation.dialogues.pkl', 'rb') as f:
    valid_data = pickle.load(f)

with open('data/hred/Test.dialogues.pkl', 'rb') as f:
    test_data = pickle.load(f)


def chunk_dialogue(dialogue):
    turns = []
    curr_turn = []

    for i, w in enumerate(dialogue):
        curr_turn.append(w)

        if w == 1 or i == len(dialogue)-1:
            if w != 1:  # In the case when last word in dialogue is not <eot>
                curr_turn.append(1)
            turns.append(curr_turn)
            curr_turn = []

    return turns


def preprocess(data):
    # Default is zero => pad token
    batch = np.zeros([args.max_turns, args.max_seq_len, len(data)], dtype=int)

    for i in tqdm.trange(len(data)):
        turns = chunk_dialogue(data[i])[:args.max_turns]

        for t in range(args.max_turns):
            seq_len = min(args.max_seq_len, len(turns[t]))
            batch[t, :seq_len, i] = turns[t][:seq_len]

    return batch


train = preprocess(train_data)
valid = preprocess(valid_data)
test = preprocess(test_data)

with open('data/hred/train.pkl', 'wb') as f:
    pickle.dump(train, f)

with open('data/hred/valid.pkl', 'wb') as f:
    pickle.dump(valid, f)

with open('data/hred/test.pkl', 'wb') as f:
    pickle.dump(test, f)
