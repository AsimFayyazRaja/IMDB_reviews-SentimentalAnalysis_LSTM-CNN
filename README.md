# IMDB_reviews-SentimentalAnalysis_LSTM-CNN
Sentiment Analysis of IMDB_reviews using LSTM &amp; ConvNets using word2vec from gensim to make feature vectors

## Usage
- Download the imdb reviews dataset from [Stanford Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) and put in the folder "data"
- Execute "getdata.py" to read data.
- Execute "getdata2.py" to get features, labels in the proper shape and make pickles.
- Execute "make_word2vecs.py" to make word2vec model.
- Run "pad_features.py" to pad them to make all features of length 128.
- Run "lstm.py" for lstm network and "conv.py" for CNN

## Results
- Got 98% training and 88% testing accuracy on simple lstm network.
- CNN didn't work well so got 60% and 53% accuracy on CNN.

## Requirements
- Python3
- Keras
- gensim(for word2vec)

## License
- It is a free public tool to use
