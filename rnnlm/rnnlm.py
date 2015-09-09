from collections import OrderedDict
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
        """Creates the neural network architecture.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Create the layers.
        self.projection_layer = ProjectionLayer(options)
        if options['encoder'] == 'lstm':
            self.encoder_layer = LSTMLayer(options)
        self.ff_layer = FFLayer(options)

        # Create the parameters.
        self.params = OrderedDict()
        self.params.update(self.projection_layer.params)
        self.params.update(self.encoder_layer.params)
        self.params.update(self.ff_layer.params)
 
        # Create Theano shared variables.
        self.theano_params = OrderedDict()
        for name, value in self.params.iteritems():
            self.theano_params[name] = theano.shared(value, name)

