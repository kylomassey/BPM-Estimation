import statistics
import math
import librosa
import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/PC/Desktop/bpm-estimator/music/Max_Libra_Horns.mp3"
y, sr = librosa.load(path, sr=None)
framel =  sr * .05
frame = librosa.util.frame(x = y, axis = 0, frame_length = int(framel), hop_length = int(framel * .5) )
window = np.hanning(frame.shape[0])
rmframe = []
for i in range(frame.shape[1]):
    windowed = frame[:,i] * window
    rmframe.append(math.sqrt(statistics.mean([sample ** 2 for sample in windowed])))
time = np.arange(len(rmframe)) * ((framel*.5)/sr)
ypoints = np.array(rmframe)
plt.plot(time, ypoints)
plt.show()