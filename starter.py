import pickle, gzip, numpy
from random_forest_classifier import random_forest_classifier

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()


random_forest_classifier(train_set, valid_set, test_set)
