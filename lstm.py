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
from keras.layers import Input, Dense, Flatten

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

#getting data
with open('padded_features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

labels=np.array(labels[:9000])

print(features.shape)
print(labels.shape)

X_train,y_train=features[:4500],labels[:4500]
X_test,y_test=features[4500:],labels[4500:]

#lstm network
input = Input(shape=(128,128))
lstm=LSTM(16,return_sequences=True)(input)
flat=Flatten()(lstm)
d=Dense(128, activation='relu')(flat)
d=Dense(256, activation='relu')(d)
output = Dense(2, activation='sigmoid')(d)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=16, epochs=15,validation_data=(X_test,y_test))

model.save('lstm.h5')
