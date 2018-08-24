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
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input, Dense, Flatten

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences

#getting data
with open('padded_features', 'rb') as fp:
    features=pickle.load(fp)

with open('labels', 'rb') as fp:
    labels=pickle.load(fp)

labels=np.array(labels[:5000])

print(labels.shape)

features=np.expand_dims(features,axis=3)

print(features.shape)

X_train,y_train=features[:4500],labels[:4500]
X_test,y_test=features[4500:],labels[4500:]

#lstm network
input = Input(shape=(128,128,1))

conv=Conv2D(8,kernel_size=5,strides=1,activation='relu')(input)
conv=MaxPooling2D(pool_size=(2, 2))(conv)

conv=Conv2D(16,kernel_size=5,strides=1,activation='relu')(conv)
conv=MaxPooling2D(pool_size=(2, 2))(conv)

conv=Conv2D(32,kernel_size=5,strides=1,activation='relu')(conv)
conv=MaxPooling2D(pool_size=(2, 2))(conv)

flat=Flatten()(conv)
d=Dense(32, activation="relu")(flat)
d=Dense(64, activation='relu')(d)
output = Dense(2, activation='sigmoid')(d)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=16, epochs=10,validation_data=(X_test,y_test))

model.save('conv.h5')
