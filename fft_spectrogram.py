import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy

path = "C:/Users/PC/Desktop/bpm-estimator/music/tone400hz.mp3"
y, sr = librosa.load(path, sr=None)
print(sr)
framel =  sr * .05
frame = librosa.util.frame(x = y, axis = 0, frame_length = int(framel), hop_length = int(framel * .5) )
window = numpy.hanning(frame.shape[0])
windowed = frame[:,11] * window
bins = librosa.fft_frequencies(sr=sr, n_fft=len(windowed))
oneframefft = numpy.fft.rfft(windowed)
mag = []
phase = []
for num in oneframefft:
    mag.append(numpy.sqrt(numpy.real(num)**2 + numpy.imag(num)**2))
    phase.append(math.atan2(numpy.imag(num), numpy.real(num)))
#plt.plot(bins,mag)
spectrum = []
for f in range(len(frame[0])):
    windowed = frame[:,f] * window
    fft = numpy.fft.rfft(windowed)
    spectrum.append(numpy.abs(fft) **2)
spectrum = numpy.array(spectrum).T
fft_db = 20 * numpy.log10(spectrum/numpy.max(spectrum) + 1e-10)
fft_db = numpy.maximum(fft_db,-60)
time = numpy.arange(len(spectrum)) * ((framel*.5)/sr)
plt.imshow(
    fft_db,
    origin='lower',
    aspect='auto',
    cmap='magma'
)
plt.colorbar(label='dB')
plt.xlabel("Time frames")
plt.ylabel("Frequency bins")
plt.show()