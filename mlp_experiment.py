import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
import timeit

import hiddenlayer

traindata_filename = 'data/cifar-100-python/train'
batch_size = 20
n_hidden = 225
n_epochs = 10
learning_rate=0.01
L1_reg=0.00
L2_reg=0.0001

def load_data(filename):
    tmp = pickle.load(open(filename))
    return (tmp['data'], tmp['fine_labels'])


def lift(data):
    return theano.shared(np.asarray(data,
                                    dtype=theano.config.floatX),
                         borrow=True)

def do_stuff():
    train_set_x, train_set_y = load_data(traindata_filename)
    train_set_x = lift(train_set_x)
    train_set_y = lift(train_set_y)
    # needs to be float on GPU, but int when comparing labels and predictions
    train_set_y = T.cast(train_set_y, 'int32')

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of
                         # [int] labels

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = hiddenlayer.MLP(
        rng=rng,
        inpt=x,
        n_in=32 * 32,
        n_hidden=n_hidden,
        n_out=100
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = timeit.default_timer()

    for epoch in range(n_epochs):
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            print('minibatch_avg_cost: {}'.format(minibatch_avg_cost))

    end_time = timeit.default_timer()
    print(('Optimization complete.'))
    print(' ran for %.2fm' % ((end_time - start_time) / 60.))

do_stuff()
