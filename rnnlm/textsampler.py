

import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class TextSampler(object):
    """Neural network language model sampler

    A Theano function that generates text using a neural network language
    model.
    """

    def __init__(self, network, options):
        """Creates the neural network architecture.

        :type network: RNNLM
        :param network: the neural network object

        :type options: dict
        :param options: a dictionary of training options
        """

        self.options = options
        self.trng = RandomStreams(1234)


        # Create the network structure.

        # x: 1 x 1
        # y is the previous word in 1-of-V encoding.
        y = tensor.vector('y_sampler', dtype='int64')
        init_state = tensor.matrix('init_state', dtype='float32')

        # A negative y value indicates this is the first word. In that case emb
        # should be a zero vector.
        emb = tensor.switch(y[:,None] < 0, 
                            tensor.alloc(0., 1, network.theano_params['Wemb'].shape[1]), 
                            network.theano_params['Wemb'][y])

        proj = network.encoder_layer.create_structure(network.theano_params,
                                                      emb,
                                                      mask=None, 
                                                      one_step=True,
                                                      init_state=init_state)
        logit = network.ff_layer.create_structure(network.theano_params,
                                                  proj,
                                                  emb,
                                                  activ='lambda x: x')
        next_probs = tensor.nnet.softmax(logit)
        next_sample = self.trng.multinomial(pvals=next_probs).argmax(1)


        # Compile the Theano function.

        # next word probability
        inps = [y, init_state]
        outs = [next_probs, next_sample, proj]

        print 'Building text sampler.'
        self.function = theano.function(inps, outs, name='f_next', profile=self.options['profile'])
        print 'Done.'

    def generate(self, maxlen=30, argmax=False):
        """ Generates a text sample.
        """
        sample = []
        sample_score = 0

        next_w = -1 * numpy.ones((1,)).astype('int64')
        next_state = numpy.zeros((1, self.options['dim'])).astype('float32')

        for ii in xrange(maxlen):
            inps = [next_w, next_state]
            ret = self.function(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]

            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0,nw]
            if nw == 0:
                break

        return sample, sample_score

