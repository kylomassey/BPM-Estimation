import librosa
import numpy

def scores_to_emissions(scores):
    scores = numpy.maximum(scores, 0)
    col_sums = scores.sum(axis=0, keepdims=True) + 1e-9
    emissions = scores/col_sums
    return emissions