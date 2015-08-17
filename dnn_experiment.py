import theano
import theano.tensor as T
import numpy as np
import timeit

import hiddenlayer

#import data_cifar as dataset
import data_mnist as dataset

batch_size = 20
n_hidden = [225, 144]
max_epochs = 30
patience_epochs = 10
learning_rate=0.01
L1_reg=0.00
L2_reg=0.0001
activation=hiddenlayer.ReLU


def lift(data):
    return theano.shared(np.asarray(data,
                                    dtype=theano.config.floatX),
                         borrow=True)


def do_stuff():
    train_set_x, train_set_y = dataset.load_train_data()
    train_set_x = lift(train_set_x)
    train_set_y = lift(train_set_y)
    # needs to be float on GPU, but int when comparing labels and predictions
    train_set_y = T.cast(train_set_y, 'int32')

    valid_set_x, valid_set_y = dataset.load_validation_data()
    valid_set_x = lift(valid_set_x)
    valid_set_y = lift(valid_set_y)
    # needs to be float on GPU, but int when comparing labels and predictions
    valid_set_y = T.cast(valid_set_y, 'int32')

    test_set_x, test_set_y = dataset.load_test_data()
    test_set_x = lift(test_set_x)
    test_set_y = lift(test_set_y)
    # needs to be float on GPU, but int when comparing labels and predictions
    test_set_y = T.cast(test_set_y, 'int32')

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

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
    classifier = hiddenlayer.DNN(
        rng=rng,
        inpt=x,
        n_in=dataset.input_shape,
        n_hidden=n_hidden,
        n_out=dataset.output_shape,
        activation=activation
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

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    start_time = timeit.default_timer()
    done_looping = False
    best_valid = None
    best_test = None
    best_epoch = None

    for epoch in xrange(max_epochs):
        if done_looping:
            break

        train_costs = [train_model(i) for i
                       in xrange(n_train_batches)]
        train_score = np.mean(train_costs)
        print('epoch {} avg train cost: {}'.format(epoch, train_score))

        valid_losses = [validate_model(i) for i
                       in xrange(n_valid_batches)]
        valid_score = np.mean(valid_losses)
        print('valid score: {}'.format(valid_score))

        if best_valid is None or valid_score < best_valid:
            best_valid = valid_score
            test_losses = [test_model(i) for i
                        in xrange(n_test_batches)]
            best_test = np.mean(test_losses)
            best_epoch = epoch
        elif epoch > patience_epochs:
            break
    print('epoch {} had best valid score: {}'.format(best_epoch, best_valid))
    print('and test score: {}'.format(best_test))

    end_time = timeit.default_timer()
    print(('Optimization complete.'))
    print(' ran for %.2fm' % ((end_time - start_time) / 60.))

do_stuff()
