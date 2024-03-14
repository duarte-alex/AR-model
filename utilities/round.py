import numpy as np

def base10_min_max_nonzero(array):
    minval=np.min(array[np.nonzero(array)])
    maxval=np.max(array[np.nonzero(array)])
    logmin=np.log10(minval)
    logmax=np.log10(maxval)
    logmin=int(np.floor(logmin))
    logmax=int(np.ceil(logmax))
    minval=10**logmin
    maxval=10**logmax
    return minval,maxval
    
def round_to_1(x):
    return round(x, -int(np.floor(np.log10(abs(x)))))
