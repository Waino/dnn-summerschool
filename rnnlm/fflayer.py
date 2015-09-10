#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
from collections import OrderedDict
import theano.tensor as tensor

from paramfunctions import normalized_weight

class FFLayer(object):
    """ Feed-Forwared Layer
    """

    def __init__(self, options):
        """Initializes the parameters for a feed-forward layer of a neural
        network.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Create the parameters.
        self.init_params = OrderedDict()

        nin = options['dim']
        nout = options['dim_word']
        self.init_params['ff_logit_lstm_W'] = normalized_weight(nin, nout, scale=0.01, ortho=False)
        self.init_params['ff_logit_lstm_b'] = numpy.zeros((nout,)).astype('float32')

        nin = options['dim_word']
        nout = options['dim_word']
        self.init_params['ff_logit_prev_W'] = normalized_weight(nin, nout, scale=0.01, ortho=False)
        self.init_params['ff_logit_prev_b'] = numpy.zeros((nout,)).astype('float32')

        nin = options['dim_word']
        nout = options['n_words']
        self.init_params['ff_logit_W'] = normalized_weight(nin, nout, scale=0.01, ortho=True)
        self.init_params['ff_logit_b'] = numpy.zeros((nout,)).astype('float32')

    def create_structure(self, theano_params, state_below, emb,
                         activ='lambda x: tensor.tanh(x)'):
        """ Creates the feed-forward layer structure.

        :type theano_params: dict
        :param theano_params: shared Theano variables

        :type state_below: theano.tensor.var.TensorVariable
        :param state_below: symbolic matrix that describes the output of the
        previous layer

        :rtype: theano.tensor.var.TensorVariable
        :returns: symbolic matrix that describes the output of this layer
        """

        logit_lstm = eval(activ)(tensor.dot(state_below, theano_params['ff_logit_lstm_W']) + theano_params['ff_logit_lstm_b'])
        logit_prev = eval(activ)(tensor.dot(emb, theano_params['ff_logit_prev_W']) + theano_params['ff_logit_prev_b'])
        logit = tensor.tanh(logit_lstm + logit_prev)
        logit = eval(activ)(tensor.dot(logit, theano_params['ff_logit_W']) + theano_params['ff_logit_b'])
	return logit

