#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
import numpy

class ModelTrainer(object):
    """Neural Network Language Model Trainer

    A Theano function that trains a neural network language model.
    """

    def __init__(self, network, optimizer_function, options):
        """Creates the neural network architecture.

        :type network: RNNLM
        :param network: the neural network object

        :type optimizer_function: str
        :param optimizer_function: name of the optimization function (adam, adadelta, rmsprop, sgd)

        :type options: dict
        :param options: a dictionary of training options
        """

        self.network = network
        self.options = options


        # Create the network structure.

        # description string: #words x #samples
        self.x = tensor.matrix('x', dtype='int64')
        self.x_mask = tensor.matrix('x_mask', dtype='float32')

        # input
        n_timesteps = self.x.shape[0]
        n_samples = self.x.shape[1]
        emb = self.network.theano_params['Wemb'][self.x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        proj = self.network.encoder_layer.create_structure(self.network.theano_params,
                                                           emb,
                                                           mask=self.x_mask)
        logit = self.network.ff_layer.create_structure(self.network.theano_params,
                                                       proj,
                                                       emb,
                                                       activ='lambda x: x')

        logit_shp = logit.shape
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0] * logit_shp[1], logit_shp[2]]))

        # cost
        x_flat = self.x.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx])
        cost = cost.reshape([self.x.shape[0], self.x.shape[1]])
        self.cost = (cost * self.x_mask).sum(0)


	# Compile the Theano functions.

        # before any regularizer

        inps = [self.x, self.x_mask]

        print 'Building f_log_probs...',
        self.f_log_probs = theano.function(inps, self.cost, profile=self.options['profile'])
        print 'Done'

        self.cost = self.cost.mean()

        if self.options['decay_c'] > 0.:
            decay_c = theano.shared(numpy.float32(self.options['decay_c']), name='decay_c')
            weight_decay = 0.
            for kk, vv in self.network.theano_params.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            self.cost += weight_decay

        # after any regularizer

        print 'Building f_cost.'
        f_cost = theano.function(inps, self.cost, profile=self.options['profile'])
        print 'Done.'

        print 'Computing gradient.'
        grads = tensor.grad(self.cost, wrt=self.__param_values())
        print 'Done'

        print 'Building f_grad.'
        f_grad = theano.function(inps, grads, profile=self.options['profile'])
        print 'Done'

        lr = tensor.scalar(name='lr')

        print 'Building optimizers.'
        if optimizer_function == 'adam':
            self.__adam(lr, grads, inps, self.cost)
        elif optimizer_function == 'adadelta':
            self.__adadelta(lr, grads, inps, self.cost)
        elif optimizer_function == 'rmsprop':
            self.__rmsprop(lr, grads, inps, self.cost)
        elif optimizer_function == 'sgd':
            self.__sgd(lr, grads, inps, self.cost)
        else:
            raise ValueError("Invalid optimizer function: %s" % optimizer_function)
        print 'Done.'


    def __param_values(self):
        """ Returns a list of the parameter values.

        self.network.theano_params is an OrderedDict.
        """
        return [value for key, value in self.network.theano_params.iteritems()]

    # optimizer functions
    # signature: __name(self, hyperp, grads, inputs (list), cost)
    # modifies: self.f_grad_shared, self.f_update

    def __adam(self, lr, grads, inp, cost):
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                   for k, p in self.network.theano_params.iteritems()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        self.f_grad_shared = theano.function(inp, cost, updates=gsup, profile=self.options['profile'])

        lr0 = 0.0002
        b1 = 0.1
        b2 = 0.001
        e = 1e-8

        updates = []

        i = theano.shared(numpy.float32(0.))
        i_t = i + 1.
        fix1 = 1. - b1**(i_t)
        fix2 = 1. - b2**(i_t)
        lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

        for p, g in zip(self.network.theano_params.values(), gshared):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (tensor.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))

        self.f_update = theano.function([lr], [], updates=updates, 
                                        on_unused_input='ignore', profile=self.options['profile'])

    def __adadelta(self, lr, grads, inp, cost):
        zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) 
                        for k, p in self.network.theano_params.iteritems()]
        running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) 
                       for k, p in self.network.theano_params.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) 
                          for k, p in self.network.theano_params.iteritems()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

        self.f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=options['profile'])
    
        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(self.__param_values(), updir)]

        self.f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=options['profile'])

    def __rmsprop(self, lr, grads, inp, cost):
        zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in self.network.theano_params.iteritems()]
        running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in self.network.theano_params.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in self.network.theano_params.iteritems()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

        self.f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=options['profile'])

        updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in self.network.theano_params.iteritems()]
        updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
        param_up = [(p, p + udn[1]) for p, udn in zip(self.__param_values(), updir_new)]
        self.f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=options['profile'])

    def __sgd(self, lr, grads, x, mask, y, cost):
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in self.network.theano_params.iteritems()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        self.f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, profile=options['profile'])

        pup = [(p, p - lr * g) for p, g in zip(self.__param_values(), gshared)]
        self.f_update = theano.function([lr], [], updates=pup, profile=options['profile'])

