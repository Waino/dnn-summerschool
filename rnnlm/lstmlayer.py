#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor

from paramfunctions import orthogonal_weight, normalized_weight

class LSTMLayer(object):
    """Long Short-Term Memory Layer
    """

    def __init__(self, options):
        """Initializes the parameters for a LSTM layer of a recurrent neural network.

        :type options: dict
        :param options: a dictionary of training options
        """

        self.options = options

        # Initialize the parameters.
        self.init_params = OrderedDict()

        nin = self.options['dim_word']
        dim = self.options['dim']
        W = numpy.concatenate([normalized_weight(nin, dim),
                               normalized_weight(nin, dim),
                               normalized_weight(nin, dim)],
                              axis=1)
        self.init_params['encoder_W'] = W

        n_gates = 3
        self.init_params['encoder_b'] = numpy.zeros((n_gates * dim,)).astype('float32')

        U = numpy.concatenate([orthogonal_weight(dim),
                               orthogonal_weight(dim),
                               orthogonal_weight(dim)],
                              axis=1)
        self.init_params['encoder_U'] = U

        Wx = normalized_weight(nin, dim)
        self.init_params['encoder_Wx'] = Wx

        Ux = orthogonal_weight(dim)
        self.init_params['encoder_Ux'] = Ux
        self.init_params['encoder_bx'] = numpy.zeros((dim,)).astype('float32')

    def create_structure(self, theano_params, state_below,
                         mask=None, one_step=False, init_state=None):
        """Creates the LSTM layer structure.

        :type theano_params: dict
        :param theano_params: shared Theano variables

        :type state_below: theano.tensor.var.TensorVariable
        :param state_below: symbolic matrix that describes the output of the
        previous layer
        """

        if one_step:
            assert init_state, 'previous state must be provided'

        nsteps = state_below.shape[0]

        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = state_below.shape[0]

        dim = theano_params['encoder_Ux'].shape[1]

        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        state_below_ = tensor.dot(state_below, theano_params['encoder_W']) + theano_params['encoder_b']
        state_belowx = tensor.dot(state_below, theano_params['encoder_Wx']) + theano_params['encoder_bx']
        U = theano_params['encoder_U']
        Ux = theano_params['encoder_Ux']

        def _step_slice(m_, x_, xx_, h_, h_out_, U, Ux):
            """ The step function for theano.scan().

            m_     -- mask
            x_     -- state_below
            xx_    -- state_belowx
            h_     -- h at prev time step
            h_out_ -- ignored
            U      -- U  (concatenated Us)
            Ux     -- Ux
            """
            preact = tensor.dot(h_, U)
            preact += x_

            i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            o = tensor.nnet.sigmoid(_slice(preact, 2, dim))

            preactx = tensor.dot(h_, Ux)
            # no longer applying r gate
            preactx = preactx + xx_

            h = tensor.tanh(preactx)

            #h = u * h_ + (1. - u) * h  # GRU
            h = f * h_ + i * h          # LSTM
            h_out = o * tensor.tanh(h)
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            h_out = m_[:,None] * h_out + (1. - m_)[:,None] * h_out_

            return h, h_out #, r, u, preact, preactx

        seqs = [mask, state_below_, state_belowx]
        shared_vars = [theano_params['encoder_U'], 
                       theano_params['encoder_Ux']]

        if init_state is None:
            init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

        if one_step:
            # When generating text, we perform just one step. Concatenate the
            # parameter arrays and call _step_slice() manually.
            rval = _step_slice(*(seqs + [init_state, init_state] + shared_vars))
        else:
            rval, updates = theano.scan(_step_slice,
                                        sequences=seqs,
                                        outputs_info = [init_state, init_state],
                                        non_sequences = shared_vars,
                                        name='encoder_layers',
                                        n_steps=nsteps,
                                        profile=self.options['profile'],
                                        strict=True)
        rval = rval[1]
        return rval

