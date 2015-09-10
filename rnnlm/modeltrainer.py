import theano.tensor as tensor

class ModelTrainer(object):
    """Neural Network Language Model Trainer

    A Theano function that trains a neural network language model.
    """

    def __init__(self, network, options):
        """Creates the neural network architecture.

        :type network: RNNLM
        :param network: the neural network object

        :type options: dict
        :param options: a dictionary of training options
        """

        opt_ret = dict()

        # description string: #words x #samples
        self.x = tensor.matrix('x', dtype='int64')
        self.x_mask = tensor.matrix('x_mask', dtype='float32')

        # input
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
        emb = network.theano_params['Wemb'][self.x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        proj = network.encoder_layer.get(network.theano_params,
                                         emb,
                                         mask=self.x_mask)
        # Unwraps the whole layer output.
        proj_h = proj[0]

        # compute word probabilities
        logit = network.ff_layer.get(network.theano_params, proj_h, emb, activ='lambda x: x')

        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))

        # cost
        x_flat = self.x.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx])
        cost = cost.reshape([self.x.shape[0], self.x.shape[1]])
        self.cost = (cost * self.x_mask).sum(0)

