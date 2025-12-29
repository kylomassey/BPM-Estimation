import statistics
import math
import librosa

path = "C:/Users/PC/Desktop/bpm-estimator/music/Max_Libra_Horns.mp3"
y, sr = librosa.load(path, sr=None)
duration = librosa.get_duration(y=y,sr=sr)
maxamp = max(abs(min(y)),max(y))
numsamples = len(y)
rms = math.sqrt(statistics.mean([sample ** 2 for sample in y]))
print("sample rate: ", sr)
print("number of samples: ", numsamples)
print("duration: ", numsamples/sr)
print("max amplitude: ", maxamp)
print("rms: ", rms)