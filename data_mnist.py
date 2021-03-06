import cPickle as pickle

data_dir = 'data/'
input_shape = 28 * 28
output_shape = 10


_whole_package = pickle.load(open('{}mnist.pkl'.format(data_dir)))


def load_train_data():
    return _whole_package[0]


def load_validation_data():
    return _whole_package[1]


def load_test_data():
    return _whole_package[2]
