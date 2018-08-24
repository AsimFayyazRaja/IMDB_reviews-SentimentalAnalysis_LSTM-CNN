import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from matplotlib import style
import string
from collections import Counter
import sys
import pickle
import glob
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Dense

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

#getting data
with open('features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

features=features[:5000]
labels=labels[:5000]

word2vecs=[]
model =Word2Vec.load('word2vec.model')

with tqdm(total=len(features)) as pbar:
    for sentence in features:
        #print(sentence)
        sublist=[]
        for word in sentence:
            try:
                x=np.array(model.wv[word])
                sublist.append(x)
            except:
                #print("exception in making word2vec of word: ", word)
                pass
        if sublist!=[]:
            word2vecs.append(np.array(sublist))
        pbar.update(1)

word2vecs=np.array(word2vecs)
print(word2vecs.shape)

with open('word2vecs', 'wb') as fp:
    pickle.dump(word2vecs, fp)

