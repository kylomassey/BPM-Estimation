import numpy
from scipy import signal

def median_smoothing(chroma, filt_len = 41): 
    assert filt_len % 2 == 1

    filt_len = [1,filt_len]
    smooth_chroma = signal.medfilt2d(chroma, filt_len)
    return smooth_chroma

def diagonal_smoothing(ssm, filt_len = 20):
    N = ssm.shape[0]
    M = ssm.shape[1]

    diag_chroma = numpy.zeros((N, M))
    extend_chroma = numpy.zeros((N + filt_len, M + filt_len))
    extend_chroma[0:N, 0:M] = ssm

    for pos in range(0, filt_len):
        diag_chroma = diag_chroma + extend_chroma[pos:(N + pos), pos:(M + pos)]
    diag_chroma = diag_chroma / filt_len
    return diag_chroma
