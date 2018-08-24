import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing, cross_validation
from tqdm import tqdm
import csv
from matplotlib import style
import string
from collections import Counter
import sys
import pickle
import glob
from tqdm import tqdm
from gensim import corpora
import gensim
from gensim.models import Word2Vec


#reading files

print("Reading all files and saving data..")

def getdata(folder,label,data):
    r=0
    with tqdm(total=len(glob.glob(folder+"/*.txt"))) as pbar:
        for files in glob.glob(folder+"/*.txt"):
            temp=[]
            f = open(files, 'r')
            x = str(f.readline())
            temp=x.split()
            temp.append(label)
            #temp=np.array(temp)
            data.append(temp)
            pbar.update(1)
            r+=1
    return data

labels=[]
sentences=[]
data=[]
folder="data/train/pos"
label=[1,0]
data=getdata(folder,label,data)

folder="data/train/neg"
label=[0,1]
data=getdata(folder,label,data)
'''
#print(sentences)
model = Word2Vec(sentences, size=128, window=4, min_count=3, workers=2)
print(model.wv['movie'])

print(model.wv.similarity('great', 'Robert'))


print("saving data..")

model.save("word2vec.model")
'''
with open('alldata', 'wb') as fp:
    pickle.dump(data, fp)

'''
with open('sentences', 'wb') as fp:
    pickle.dump(sentences, fp)

with open('labels', 'wb') as fp:
    pickle.dump(labels, fp)
'''
print("Data saved.")
'''
print(len(sentences))
print(len(labels))
'''