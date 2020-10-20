import numpy as np

def normalize(v, p=2):
    ''' project vector on to unit L-p ball. '''
    norm=np.linalg.norm(v, ord=p)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def sqexp_p_vectors(x, y, p=2, scale=1.0):
    ''' compute rbf kernel between two vectors x and y '''
    
    d = np.sum(np.power(np.abs(x - y), p), axis=1)
    d = np.expand_dims(np.exp(-d), axis=1)

    return d