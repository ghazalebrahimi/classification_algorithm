import pickle, gzip, numpy

from ada_boost_classifier import ada_boost_classifier
from gradient_boosting_classifier import gradient_boosting_classifier
from linear_model_classifiers import linear_model_classifiers
from random_forest_classifier import random_forest_classifier
from support_vector_classifiers import support_vector_classifiers

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

linear_model_classifiers(train_set, valid_set, test_set)
support_vector_classifiers(train_set, valid_set, test_set)
ada_boost_classifier(train_set, valid_set, test_set)
random_forest_classifier(train_set, valid_set, test_set)
gradient_boosting_classifier(train_set, valid_set, test_set)

