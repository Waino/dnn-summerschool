import numpy
from collections import OrderedDict

from paramfunctions import normalized_weight

class FFLayer(object):
    def __init__(self, options):
        """
        Feed-forward layer of a neural network.

        Currently just initializes the parameters.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Create the parameters.
        self.params = OrderedDict()

        nin = options['dim']
        nout = options['dim_word']
        self.params['ff_logit_lstm_W'] = normalized_weight(nin, nout, scale=0.01, ortho=False)
        self.params['ff_logit_lstm_b'] = numpy.zeros((nout,)).astype('float32')

        nin = options['dim_word']
        nout = options['dim_word']
        self.params['ff_logit_prev_W'] = normalized_weight(nin, nout, scale=0.01, ortho=False)
        self.params['ff_logit_prev_b'] = numpy.zeros((nout,)).astype('float32')

        nin = options['dim_word']
        nout = options['n_words']
        self.params['ff_logit_W'] = normalized_weight(nin, nout, scale=0.01, ortho=True)
        self.params['ff_logit_b'] = numpy.zeros((nout,)).astype('float32')

