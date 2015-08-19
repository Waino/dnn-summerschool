'''
Build a simple neural machine translation model
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

from data_iterator import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
            state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
            state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive'%kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'), 
          'gru': ('param_init_gru', 'gru_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def linear(x):
    return x

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

# batch preparation
def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.

    return x, x_mask

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    # Waino: big-U is concatenation of all U:s, later sliced
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, options, prefix='gru', 
              mask=None, one_step=False, init_state=None, **kwargs):
    # Waino: state_below: x from the slides (embedding of previous word)
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # Waino: these are known at train time without needing to scan,
    # Waino: (x:s are known, and W:s are constant during the minibatch)
    # Waino: and thus can be calculated for all words in one operation
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        # Waino: m_ = mask
        # Waino: x_ = state_below_ (concatenation of Wr * x + br, Wu * x + bu)
        # Waino: xx_= state_belowx (W * x + b)
        # Waino: h_ = init_state
        # Waino: U  = U  (concatenated Ur, Uu)
        # Waino: Ux = Ux
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # Waino: maybe Ux is just U on the slides?
        # Waino: elementwise mult can be moved outside the matrix mult
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')], 
                     tparams[_p(prefix, 'Ux')]]

    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state],
                                    non_sequences = shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval]
    return rval

# LSTM layer (by Waino)
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),   # one more gate
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', 
              mask=None, one_step=False, init_state=None, **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = state_below.shape[0]

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        # Waino: m_ = mask
        # Waino: x_ = state_below
        # Waino: xx_= state_belowx
        # Waino: h_ = init_state
        # Waino: U  = U  (concatenated Us)
        # Waino: Ux = Ux
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')], 
                     tparams[_p(prefix, 'Ux')]]

    if init_state is None:
        init_state = tensor.unbroadcast(tensor.alloc(0., n_samples, dim), 0)

    if one_step:
        rval = _step(*(seqs+[init_state]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info = [init_state],
                                    non_sequences = shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    rval = [rval]
    return rval

# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder', 
                                              nin=options['dim_word'], 
                                              dim=options['dim'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', 
                                nin=options['dim'], nout=options['dim_word'], 
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev', 
                                nin=options['dim_word'], nout=options['dim_word'], 
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit', 
                                nin=options['dim_word'], nout=options['n_words'])

    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # input
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask)
    proj_h = proj[0]

    # compute word probabilities
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options, 
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options, 
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0],x.shape[1]])
    cost = (cost * x_mask).sum(0)

    return trng, use_noise, x, x_mask, opt_ret, cost

# build a sampler
def build_sampler(tparams, options, trng):
    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero
    emb = tensor.switch(y[:,None] < 0, 
                        tensor.alloc(0., 1, tparams['Wemb'].shape[1]), 
                        tparams['Wemb'][y])
    proj = get_layer(options['encoder'])[1](tparams, emb, options, 
                                            prefix='encoder', 
                                            mask=None, 
                                            one_step=True, 
                                            init_state=init_state)
    next_state = proj[0]

    logit_lstm = get_layer('ff')[1](tparams, next_state, options, 
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options, 
                                    prefix='ff_logit_prev', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev)
    logit = get_layer('ff')[1](tparams, logit, options, 
                               prefix='ff_logit', activ='linear')
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    print 'Building f_next..', 
    inps = [y, init_state]
    outs = [next_probs, next_sample, next_state]
    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print 'Done'

    return f_next

# generate sample
def gen_sample(tparams, f_next, options, trng=None, maxlen=30, argmax=False):

    sample = []
    sample_score = 0

    next_w = -1 * numpy.ones((1,)).astype('int64')
    next_state = numpy.zeros((1,options['dim'])).astype('float32')

    for ii in xrange(maxlen):
        inps = [next_w, next_state]
        ret = f_next(*inps)
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

def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x in iterator:
        n_done += len(x)

        x, x_mask = prepare_data(x, n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            import ipdb; ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed'%(n_done)

    return numpy.array(probs)

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0., 
                             name='%s_grad'%k) 
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

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

    for p, g in zip(tparams.values(), gshared):
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

    f_update = theano.function([lr], [], updates=updates, 
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), 
                                  name='%s_grad'%k) 
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), 
                                 name='%s_rup2'%k) 
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), 
                                    name='%s_rgrad2'%k) 
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=profile)
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, mask, y, cost):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(dim_word=100, # word vector dimensionality
          dim=1000, # the number of LSTM units
          encoder='gru',
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0., 
          alpha_c=0., 
          diag_c=0.,
          lrate=0.01, 
          n_words=100000,
          maxlen=100, # maximum length of the description
          optimizer='rmsprop', 
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          sampleFreq=100, # generate some samples after every sampleFreq updates
          dataset='../data/morph/finnish.clean.train',
          valid_dataset='../data/morph/finnish.clean.test',
          dictionary='../data/morph/morph.vocab',
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    worddicts = dict()
    worddicts_r = dict()
    with open(dictionary, 'rb') as f:
        for (i, line) in enumerate(f):
            word = line.strip()
            code = i + 2
            worddicts_r[code] = word
            worddicts[word] = code

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(dataset, 
                         worddicts,
                         n_words_source=n_words, 
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_dataset, 
                         worddicts,
                         n_words_source=n_words, 
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, use_noise, \
          x, x_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask]

    print 'Buliding sampler'
    f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    print 'Building f_grad...',
    f_grad = theano.function(inps, grads, profile=profile)
    print 'Done'

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask = prepare_data(x, maxlen=maxlen, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                #import ipdb; ipdb.set_trace()

                if best_p != None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(5):
                    sample, score = gen_sample(tparams, f_next, 
                                               model_options, trng=trng, 
                                               maxlen=30, argmax=False)
                    print 'Sample ', jj, ': ',
                    ss = sample
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r:
                            print worddicts_r[vv], 
                        else:
                            print 'UNK',
                    print

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data, model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    import ipdb; ipdb.set_trace()

                print 'Valid ', valid_err

        print 'Seen %d samples'%n_samples

        if estop:
            break

    if best_p is not None: 
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, 
                history_errs=history_errs, 
                **params)

    return valid_err


if __name__ == '__main__':
    pass












    


