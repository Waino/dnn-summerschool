#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

from paramfunctions import normalized_weight

class ProjectionLayer(object):
    """Projection Layer for Neural Network Language Model
    """

    def __init__(self, options):
        """Initializes the parameters for the first layer of a neural network
        language model, which creates the word embeddings.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Initialize the parameters.
        self.init_params = OrderedDict()

        nin = options['n_words']
        nout = options['dim_word']
        self.init_params['Wemb'] = normalized_weight(nin, nout)

