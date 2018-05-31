# Domain Knowledge enhanced Gated recurrent Unit (DKE-GRU)

Project for running DKE-GRU model.

## Getting Started

Install the requirements.txt file and install pytorch version: "0.3.1.post2"

### Prerequisites

Download the pre-processed files from Wu et. al, from here: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu%20data.zip?dl=0
Use the vocab.txt to create a list of three dictionaries train, valid and test and save it as a pickle called dataset.pkl and save it in ubuntu_data.
The dictionaries should have the keys c, r and y denoting context response and label. The word must be replaced with word_id from the vocab dictionary.

This will be read from data.py
Use the train.txt file to train a fasttext model using the fasttext library:https://github.com/facebookresearch/fastText by:
./fasttext skipgram -input train.txt -dim 200 -output fast_text_200

Save this file into a numpy array whose index corresponds to the word_id from the previous dictionary and the row contains the fasttext vector for that word.
copy the file to ubuntu_data directory.

Download the ubuntu_description.npy file provided and copy it to ubuntu_data directory

## Running the model

The DKE-GRU model should be run as:


