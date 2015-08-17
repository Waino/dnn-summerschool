import theano
import theano.tensor as T
import numpy as np

import cPickle as pickle

traindata_filename = 'data/cifar-100-python/train'

def load_data(filename):
    return pickle.load(open(filename))
