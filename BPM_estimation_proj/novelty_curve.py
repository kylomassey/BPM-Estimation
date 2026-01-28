import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy

def onset(spectrum, hop_time):
    spectrum =  numpy.mean(spectrum, axis=0)
    return numpy.maximum(numpy.diff(spectrum)/hop_time, 0)

def smooth(onset, frame_len):