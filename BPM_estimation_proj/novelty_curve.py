import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy

def onset_curve(spectrum, hop_time):
    spectrum =  numpy.mean(spectrum, axis=0)
    return numpy.maximum(numpy.diff(spectrum)/hop_time, 0)

def smooth_curve(onset, frame_len):
    kernel = numpy.ones(frame_len, dtype=float)
    smooth = numpy.convolve(onset, kernel, mode="same")
    return smooth