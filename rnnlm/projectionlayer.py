from collections import OrderedDict

from paramfunctions import normalized_weight

class ProjectionLayer(object):
    def __init__(self, options):
        """
        The first layer of a neural network language model, which creates the
        word embeddings.

        Currently just initializes the parameters.

        :type options: dict
        :param options: a dictionary of training options
        """

        # Create the parameters.
        self.params = OrderedDict()

        nin = options['n_words']
        nout = options['dim_word']
        self.params['Wemb'] = normalized_weight(nin, nout)

