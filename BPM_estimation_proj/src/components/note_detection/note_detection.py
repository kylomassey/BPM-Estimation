import librosa
import numpy
from .hmm_viterbi import scores_to_emissions, viterbi
from .detection_tools import note_detection, match_chord
from .adjustments import median_smoothing, diagonal_smoothing, downsample_time
from ..spectrogram import spectrogram
from ..frequency_ranges import freq_range
from ..visualization import display_spectrogram, display_chromagram, display_ssm, display_chords, display_viterbi
from .ssm import self_similarity_matrix


def stft_chord_analyzer(path, filename):
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
    print("spectrogram done")
    display_spectrogram(spectrum, hop_time, filename, bin_size)

    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)

    #start freq must be a multiple of the bin_size for accuracy
    sheet = note_detection(spectrum.full_range[round(bin_size*frame_len/sample_rate):], bin_size, start_freq=bin_size)
    print("chromagram done")
    display_chromagram(sheet, filename)

    chord_analysis, max_chord_analysis, chord_string, labels = match_chord(sheet)
    print("chordogram done")

    display_chords(max_chord_analysis, numpy.array(labels), filename)

    #sheet = median_smoothing(sheet, 99)    

    factor = 20
    sheet = downsample_time(sheet, factor)

    sheet = self_similarity_matrix(sheet)

    sheet = diagonal_smoothing(sheet, 20)

    display_ssm(sheet, hop_time, filename, factor)

    choice = input("Would you like to analyze another song? (y/n): ")
    choice = choice.lower()
    if choice == 'y':
        return True
    else:
        print("Thank you for using the BPM estimator!")
        return False

def cqt_chord_analyzer(path, filename):
    try:
        y, sample_rate = librosa.load(path, sr=None)
    except FileNotFoundError:
        print("File not found. Please check the file name and try again.")
        return True
    print(sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sample_rate)
    fmin = librosa.note_to_hz('C1') * 2**(tuning / 12)
    hop_len = 1024
    hop_time = hop_len/sample_rate

    cqt = numpy.abs(librosa.cqt(y_harmonic, hop_length=hop_len, fmin=fmin, tuning=None))

    display_spectrogram(cqt, hop_time, filename)

    chromagram = cqt.reshape(7,12,cqt.shape[1]).sum(axis=0)

    display_chromagram(chromagram, filename, hop_time)

    chord_analysis, max_chord_analysis, chord_string, labels = match_chord(chromagram)
    display_chords(max_chord_analysis, numpy.array(labels), filename)

    emissions = scores_to_emissions(chord_analysis)
    viterbi_analysis = viterbi(emissions)

    display_viterbi(emissions, viterbi_analysis, hop_time, filename)

    totals = max_chord_analysis.sum(axis=1)
    for i in numpy.argsort(totals)[::-1][:6]:
        print(f"{labels[i]:10} {totals[i]:.1f}")
    
    choice = input("Would you like to analyze another song? (y/n): ")
    choice = choice.lower()
    if choice == 'y':
        return True
    else:
        print("Thank you for using the BPM estimator!")
        return False