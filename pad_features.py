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
with open('word2vecs', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

labels=labels[:5000]

features=pad_sequences(features,maxlen=128)

print(features.shape)

with open('padded_features', 'wb') as fp:
    pickle.dump(features, fp)

'''
#lstm network
a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
'''
