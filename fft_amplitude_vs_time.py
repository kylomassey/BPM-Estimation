import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy

path = "C:/Users/PC/Desktop/bpm-estimator/music/effi.mp3"
y, sr = librosa.load(path, sr=None)
print(sr)
framel =  sr * .05
frame = librosa.util.frame(x = y, frame_length = int(framel), hop_length = int(framel * .5) )
window = numpy.hanning(frame.shape[0])
spectrum = []
for f in range(len(frame[0])):
    windowed = frame[:,f] * window
    fft = numpy.fft.rfft(windowed)
    spectrum.append(numpy.abs(fft) **2)
spectrum = numpy.array(spectrum).T
low, lowmid, highmid, high = [],[],[],[]
#print(spectrum[int((20*framel)/sr):int((200*framel)/sr),f])
for f in range(len(spectrum[0])):
    low.append(numpy.average(spectrum[int((20*framel)/sr):int((200*framel)/sr),f]))
    lowmid.append(numpy.average(spectrum[int((200*framel)/sr):int((800*framel)/sr),f]))
    highmid.append(numpy.average(spectrum[int((800*framel)/sr):int((3000*framel)/sr),f]))
    high.append(numpy.average(spectrum[int((3000*framel)/sr):int((8000*framel)/sr),f]))
print(len(spectrum))