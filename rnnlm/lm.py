'''
Build a recurrent NNLM.
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

from rnnlm import RNNLM
from modeltrainer import ModelTrainer
from textsampler import TextSampler


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

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

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
layers = {'ff': 'fflayer', 
          'gru': 'gru_layer',
          'lstm': 'lstm_layer'}

def get_layer(name):
    """ Returns the layer function that creates the layer structure. """
    return eval(layers[name])

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

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([normalized_weight(nin,dim),
                               normalized_weight(nin,dim)], axis=1)
        params[_p(prefix,'W')] = W
        params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    # Waino: big-U is concatenation of all U:s, later sliced
    U = numpy.concatenate([orthogonal_weight(dim),
                           orthogonal_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = normalized_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = orthogonal_weight(dim)
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
        # Waino: h_ = h at previous timestep, initialized to init_state (h_(t-1))
        # Waino: U  = U  (concatenated Ur, Uu)
        # Waino: Ux = Ux
        # Waino: h = h_t
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

        # Waino: u is negated compared to slides (doesn't matter due to symmetry)
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
                                    profile=options['profile'],
                                    strict=True)
    rval = [rval]
    return rval

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
def adam(lr, tparams, grads, inp, cost, options):
    gshared = [theano.shared(p.get_value() * 0., 
                             name='%s_grad'%k) 
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=options['profile'])

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
                               on_unused_input='ignore', profile=options['profile'])

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost, options):
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

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up, profile=options['profile'])
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=options['profile'])

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost, options):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up, profile=options['profile'])

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=options['profile'])

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, x, mask, y, cost, options):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup, profile=options['profile'])

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=options['profile'])

    return f_grad_shared, f_update


def train(dim_word=100, # word vector dimensionality
          dim=1000, # the number of LSTM units
          encoder='lstm',
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
          dataset='../data/morph/finnish.clean.train10k',
          valid_dataset='../data/morph/finnish.clean.test',
          dictionary='../data/morph/morph.vocab',
          use_dropout=False,
          reload_=False,
          profile=False):

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

    rnnlm = RNNLM(model_options)
    params = rnnlm.params
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = rnnlm.theano_params

    trng = RandomStreams(1234)
    
    trainer = ModelTrainer(rnnlm, model_options)
    inps = [trainer.x, trainer.x_mask]

    print 'Buliding text sampler.'
    sampler = TextSampler(rnnlm, model_options, trng)
    print 'Done.'

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, trainer.cost, profile=model_options['profile'])
    print 'Done'

    cost = trainer.cost.mean()

    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=model_options['profile'])
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'
    print 'Building f_grad...',
    f_grad = theano.function(inps, grads, profile=model_options['profile'])
    print 'Done'

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, model_options)
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
                    sample, score = sampler.generate()
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

    valid_err = pred_probs(f_log_probs, prepare_data, model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, 
                history_errs=history_errs, 
                **params)

    return valid_err


if __name__ == '__main__':
    pass
