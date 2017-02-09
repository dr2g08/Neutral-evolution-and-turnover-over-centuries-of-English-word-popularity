maxlist = 1000

import numpy as np
import pickle
import pandas as pd


with open('data/counts.pickle', 'rb') as f:
    counts = pickle.load(f)

# obtain the corpus size and vocab size for the ngrams dataset (a-z)
years = range(1700,2001)

corpus = []
vocab = []
for y in years:
    vocab.append(len(counts[y]))
    corpus.append(sum(counts[y]))
    
vocab = np.asarray(vocab)
corpus = np.asarray(corpus)


with open('summary/corpus.pickle', 'wb') as handle:
        pickle.dump(corpus, handle)

with open('summary/vocab.pickle', 'wb') as handle:
        pickle.dump(vocab, handle)

# save the frequency vs rank data for Ngrams data 
for y in [1700,1800,1900,2000]:
    rank_freq = np.asarray(sorted(counts[y],reverse=True))
    
    with open('summary/'+str(y)+'.pickle', 'wb') as handle:
            pickle.dump(rank_freq, handle)
