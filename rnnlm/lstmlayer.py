from collections import OrderedDict
import numpy

from paramfunctions import orthogonal_weight, normalized_weight

class LSTMLayer(object):
    def __init__(self, options):
        """
        Long short-term memory layer for a recurrent neural network.

        Currently just initializes the parameters.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Create the parameters.
        self.params = OrderedDict()

        nin=options['dim_word']
        dim=options['dim']
        W = numpy.concatenate([normalized_weight(nin, dim),
                               normalized_weight(nin, dim),
                               normalized_weight(nin, dim)],
                              axis=1)
        self.params['encoder_W'] = W

        n_gates = 3
        self.params['encoder_b'] = numpy.zeros((n_gates * dim,)).astype('float32')

        U = numpy.concatenate([orthogonal_weight(dim),
                               orthogonal_weight(dim),
                               orthogonal_weight(dim)],
                              axis=1)
        self.params['encoder_U'] = U

        Wx = normalized_weight(nin, dim)
        self.params['encoder_Wx'] = Wx

        Ux = orthogonal_weight(dim)
        self.params['encoder_Ux'] = Ux
        self.params['encoder_bx'] = numpy.zeros((dim,)).astype('float32')

