import librosa
import numpy
from .detection_tools import note_detection, match_chord
from .adjustments import median_smoothing, diagonal_smoothing, downsample_time
from ..spectrogram import spectrogram
from ..frequency_ranges import freq_range
from ..visualization import display_spectrogram, display_chromagram, display_ssm, display_chords
from .ssm import self_similarity_matrix


def chord_analyzer(path, filename):
    try:
        y, sample_rate = librosa.load(path, sr=None)
    except FileNotFoundError:
        print("File not found. Please check the file name and try again.")
        return True
    print(sample_rate)

    frame_len =  int(sample_rate * .05)
    hop_len = int(frame_len * .25)
    framed_audio = librosa.util.frame(x = y, frame_length = frame_len, hop_length = hop_len)
    bin_size = sample_rate / frame_len
    hop_time = hop_len / sample_rate
    
    spectrum = spectrogram(framed_audio)

    display_spectrogram(spectrum, hop_time, filename, bin_size)

    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)

    #start freq must be a multiple of the bin_size for accuracy
    sheet = note_detection(spectrum.full_range[round(bin_size*frame_len/sample_rate):], bin_size, start_freq=bin_size)

    display_chromagram(sheet, filename)

    chord_analysis, max_chord_analysis, chord_string, labels = match_chord(sheet)
    print(labels)

    display_chords(max_chord_analysis, numpy.array(labels), filename)

    #sheet = median_smoothing(sheet, 99)    

    factor = 20
    sheet = downsample_time(sheet, factor)

    sheet = self_similarity_matrix(sheet)

    sheet = diagonal_smoothing(sheet, 20)

    display_ssm(sheet, hop_time, filename, factor)

    print("would you like to analyze another song? (y/n)")
    choice = input().lower()
    if choice == 'y':
        return True
    else:
        print("Thank you for using the BPM estimator!")
        return False
