#Main.py
import statistics
import math
import librosa
import matplotlib.pyplot as plt
from spectrogram import spectrogram
from frequency_ranges import freq_range
from visualization import display_spectrogram, display_tempogram
from novelty_curve import process_band
from BPM_estimation import auto_correlation, harmonic_scoring, tempogram
import numpy

def main():
    #Here I establish the path of the audio to be analyzed. 
    #I use librosa.load to obtain the audio as a time series. Using the sample rate librosa found from the audio
    #I get the frame length for 50ms of audio and the hop length of 12.5 ms of audio. These values are then used
    #as parameters in librosa's frame method. It will return to me the time series as array of shape (number of frames, frame length)
    #the hop length signifies the number of frames each frame length is pushed. For example [1,2,3,4,5,6,7,8] frame lenth 4 hop length 2 returns
    #[[1,2,3,4], [3,4,5,6], [5,6,7,8]]. librosa.util.frame returns this but instead transposed for convenience 
    
    path = "./music/speed_demon.mp3"
    y, sample_rate = librosa.load(path, sr=None)
    print(sample_rate)
    frame_len =  int(sample_rate * .05)
    hop_len = int(frame_len * .25)
    framed_audio = librosa.util.frame(x = y, frame_length = frame_len, hop_length = hop_len)

    #This signifies the amount of time passed between each hop
    hop_time = hop_len / sample_rate

    #Spectrogram takes the framed values and executes the fourier transform on each frame
    #it also saves the spectrogram to the main project folder
    spectrum = spectrogram(framed_audio)
    display_spectrogram(spectrum=spectrum, hop_time=hop_time)

    #Divides up the fft spectrum into frequency ranges
    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)

    #Computes the onset curves for each frequency band
    master_curve = process_band(spectrum.full_range, hop_time)
    sub_bass_curve = process_band(spectrum.sub_bass_range, hop_time)
    bass_curve = process_band(spectrum.bass_range, hop_time)

    #Compute high and low lag based on expected bpm range calculated by dividing 60 by the hop time then the high and low bpm thresholds
    bpm_low = 60
    bpm_high = 180

    lag_low = int(60 / hop_time / bpm_high)
    lag_high = int(60 / hop_time / bpm_low)

    print("lag low/bpm high:", lag_low, "/", bpm_high, "|| lag high/bpm low:", lag_high, " / ", bpm_low)

    #Creates and array of bpm buckets correlated to each lag value within the range
    bpm_graph = numpy.divide(60, numpy.multiply(numpy.arange(lag_low, lag_high + 1),hop_time))

    #Computes autocorrelation to determine first estimated bpm
    master_correlation = auto_correlation(master_curve, lag_low, lag_high)
    sub_bass_correlation = auto_correlation(sub_bass_curve, lag_low, lag_high)
    bass_correlation = auto_correlation(bass_curve, lag_low, lag_high)

    master_correlation_bpm = bpm_graph[numpy.argmax(master_correlation)]
    sub_bass_correlation_bpm = bpm_graph[numpy.argmax(sub_bass_correlation)]
    bass_correlation_bpm = bpm_graph[numpy.argmax(bass_correlation)]

    print("master corr: ", master_correlation_bpm, "\nsub bass corr: ", sub_bass_correlation_bpm, "\nbass correlation: ", bass_correlation_bpm)

    #Computes the harmonic scoring using the auto correlation data
    master_harmonic_scoring = bpm_graph[harmonic_scoring(master_correlation, lag_low, lag_high)]
    sub_bass_harmonic_scoring = bpm_graph[harmonic_scoring(sub_bass_correlation, lag_low, lag_high)]
    bass_harmonic_scoring = bpm_graph[harmonic_scoring(bass_correlation, lag_low, lag_high)]

    print("master harm score: ", master_harmonic_scoring, "\nsub bass harm score: ", sub_bass_harmonic_scoring, "\nbass harm score", bass_harmonic_scoring)

    master_tempogram, master_highscore = tempogram(master_curve, hop_time, lag_low, lag_high)
    master_tempogram_bpm = bpm_graph[master_highscore]
    sub_bass_tempogram, sub_bass_highscore = tempogram(sub_bass_curve, hop_time, lag_low, lag_high)
    sub_bass_tempogram_bpm = bpm_graph[sub_bass_highscore]
    bass_tempogram, bass_highscore = tempogram(bass_curve, hop_time, lag_low, lag_high)
    bass_tempogram_bpm = bpm_graph[bass_highscore]

    print("master tempogram score: ", master_tempogram_bpm, "\nsub bass tempogram score: ", sub_bass_tempogram_bpm, "\n bass tempogram score: ", bass_tempogram_bpm)
    display_tempogram(master_tempogram, bpm_graph)
    

main()