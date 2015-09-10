import numpy
from collections import OrderedDict
import theano.tensor as tensor

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

    def get(self, theano_params, state_below, emb, activ='lambda x: tensor.tanh(x)'):
        logit_lstm = eval(activ)(tensor.dot(state_below, theano_params['ff_logit_lstm_W']) + theano_params['ff_logit_lstm_b'])
        logit_prev = eval(activ)(tensor.dot(emb, theano_params['ff_logit_prev_W']) + theano_params['ff_logit_prev_b'])
        logit = tensor.tanh(logit_lstm + logit_prev)
        logit = eval(activ)(tensor.dot(logit, theano_params['ff_logit_W']) + theano_params['ff_logit_b'])
	return logit

