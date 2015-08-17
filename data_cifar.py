import numpy as np
import cPickle as pickle

data_dir = 'data/cifar-10-batches-py/'
train_batches = range(1, 5)
validation_batches = [5]


def load_batches(batches):
    data = []
    labels = []
    for batch in batches:
        tmp = pickle.load(open('{}data_batch_{}'.format(data_dir, batch)))
        data.append(tmp['data'])
        labels.append(tmp['labels'])
    return (np.concatenate(data),
            np.concatenate(labels))


def load_train_data():
    return load_batches(train_batches)


def load_validation_data():
    return load_batches(validation_batches)


def load_test_data():
    tmp = pickle.load(open('{}test_batch'.format(data_dir)))
    return (tmp['data'], np.asarray(tmp['labels']))
