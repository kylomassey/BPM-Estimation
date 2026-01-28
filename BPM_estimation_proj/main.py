#Main.py
import statistics
import math
import librosa
import matplotlib.pyplot as plt
from spectrogram import spectrogram
from frequency_ranges import freq_range
import numpy

def main():
    #Here I establish the path of the audio to be analyzed. 
    #I use librosa.load to obtain the audio as a time series. Using the sample rate librosa found from the audio
    #I get the frame length for 50ms of audio and the hop length of 12.5 ms of audio. These values are then used
    #as parameters in librosa's frame method. It will return to me the time series as array of shape (number of frames, frame length)
    #the hop length signifies the number of frames each frame length is pushed. For example [1,2,3,4,5,6,7,8] frame lenth 4 hop length 2 returns
    #[[1,2,3,4], [3,4,5,6], [5,6,7,8]]. librosa.util.frame returns this but instead transposed for convenience 
    
    path = "C:/Users/PC/Desktop/bpm-estimator/music/effi.mp3"
    y, sample_rate = librosa.load(path, sr=None)
    print(sample_rate)
    frame_len =  int(sample_rate * .05)
    hop_len = int(frame_len * .25)
    framed_audio = librosa.util.frame(x = y, frame_length = frame_len, hop_length = hop_len)

    #This signifies the amount of time passed between each hop
    hop_time = hop_len / sample_rate

    #Spectrogram takes the framed values and executes the fourier transform on each frame
    #it also saves the spectrogram to the main project folder
    spectrum = spectrogram(framed_audio, hop_time)

    #Divides up the fft spectrum into frequency ranges
    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)

main()