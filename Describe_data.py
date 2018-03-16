import pandas as pd
import numpy as np
from gensim.models import tfidfmodel
from gensim.corpora import Dictionary

training_f = 'data/training.csv'
train_data = pd.read_csv(training_f)
contexts = np.array(train_data['Context'])
contexts = [sent.lower() for sent in contexts]
context_unq = set(contexts)

print len(context_unq)



