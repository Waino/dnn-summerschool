import numpy

def orthogonal_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')

def normalized_weight(nin, nout, scale=0.01, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = orthogonal_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

