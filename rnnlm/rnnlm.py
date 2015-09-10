#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import os
import numpy
import theano

from projectionlayer import ProjectionLayer
from lstmlayer import LSTMLayer
from fflayer import FFLayer

class RNNLM(object):
    """Recursive Neural Network Language Model

    A recursive neural network language model implemented using Theano.
    Supports LSTM architecture.
    """

    def __init__(self, options):
        """Initializes the neural network parameters for all layers, and
        creates Theano shared variables from them.

        :type options: dict
        :param options: a dictionary of training options
        """

        self.options = options

        # This class stores the training error history too, because it's saved
        # in the same .npz file.
        self.error_history = []

        # Create the layers.
        self.projection_layer = ProjectionLayer(self.options)
        if self.options['encoder'] == 'lstm':
            self.encoder_layer = LSTMLayer(self.options)
        self.ff_layer = FFLayer(self.options)

        # Initialize the parameters.
        self.init_params = OrderedDict()
        self.init_params.update(self.projection_layer.init_params)
        self.init_params.update(self.encoder_layer.init_params)
        self.init_params.update(self.ff_layer.init_params)

        # Reload the parameters from disk if requested.
        if self.options['reload_state'] and os.path.exists(self.options['model_path']):
            self.__load_params()
 
        # Create Theano shared variables.
        self.theano_params = OrderedDict()
        for name, value in self.init_params.iteritems():
            self.theano_params[name] = theano.shared(value, name)

    def __load_params(self):
        """Loads the neural network parameters from disk.
        """

        path = self.options['model_path']
        print("Loading previous state from %s." % path)

        # Reload the parameters.
        data = numpy.load(path)
        for name in self.init_params:
            if name not in data:
                warnings.warn('The parameter %s was not found from the archive.' % name)
                continue
            self.init_params[name] = data[name]

        # Reload the error history.
        if 'error_history' not in data:
            warnings.warn('Error history was not found from the archive.' % name)
        else:
            saved_error_history = data['error_history'].tolist()
            # If the error history was empty when the state was saved,
            # ndarray.tolist() will return None.
            if saved_error_history != None:
                self.error_history = saved_error_history

        print("Done.")

    def save_params(self, x=None):
        """Saves the neural network parameters to disk.

        :type x: dict
        :param x: if set to other than None, save these values, instead of the
                  current values from the Theano shared variables
        """

        path = self.options['model_path']
        print("Saving current state to %s." % path)

        params = x if x != None else self.get_param_values()
        numpy.savez(path, error_history=self.error_history, **params)

        print("Done.")

    def get_param_values(self):
        """Pulls parameter values from Theano shared variables.
        """

        result = OrderedDict()
        for name, param in self.theano_params.iteritems():
            result[name] = param.get_value()
        return result

    def set_param_values(self, x):
        """Sets the values of Theano shared variables.
        """

        for name, value in x.iteritems():
            self.theano_params[name].set_value(value)

