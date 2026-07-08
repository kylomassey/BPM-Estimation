import librosa
import numpy

def scores_to_emissions(scores):
    scores = numpy.maximum(scores, 0)
    col_sums = scores.sum(axis=0, keepdims=True) + 1e-9
    emissions = scores/col_sums
    return emissions

def viterbi(emissions):
    l = emissions.shape[1]
    score = numpy.empty((24,l))
    psi = numpy.empty((24, l), dtype=int)
    a = numpy.load("transmission/transition_matrix.npy")
    loga = numpy.log(a).T
    logb = numpy.log(emissions + 1e-12)
    logpi = numpy.full(24, -numpy.log(24))
    score[:,0] = logb[:,0] + logpi
    for i in range(1,l):
        x = score[:,i-1] + loga
        psi[:, i] = numpy.argmax(x, axis = 1)
        score[:, i] = numpy.max(x, axis = 1) + logb[:,i]
    path = numpy.empty(l, dtype=int)
    path[-1] = numpy.argmax(score[:, -1])
    for i in range(l - 2, -1, -1):
        path[i] = psi[:, i + 1][path[i + 1]]
    return path

