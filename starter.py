import pickle, gzip, numpy

from gradient_boosting_classifier import gradient_boosting_classifier
from random_forest_classifier import random_forest_classifier

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()


random_forest_classifier(train_set, valid_set, test_set)
gradient_boosting_classifier(train_set, valid_set, test_set)