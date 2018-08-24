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

from random import shuffle


#getting data
with open('alldata', 'rb') as fp:
    data=pickle.load(fp)

data=np.array(data)
print(data.shape)

#[shuffle(sublist) for sublist in data]

shuffle(data)
'''
print(data[0])

print(data[0][-1])
'''
features=[]
labels=[]

with tqdm(total=len(data)) as pbar:
    for d in data:
        features.append(d[:-1])
        labels.append(d[-1])
        pbar.update(1)

print(features[:2])
print(labels[:2])

print("saving data..")

features=np.array(features)
labels=np.array(labels)

print(features.shape)
print(labels.shape)

with open('features', 'wb') as fp:
    pickle.dump(features, fp)

with open('labels', 'wb') as fp:
    pickle.dump(labels, fp)

print("Data saved.")

