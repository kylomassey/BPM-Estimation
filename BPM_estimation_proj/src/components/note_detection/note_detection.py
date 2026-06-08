import librosa
import numpy
from ..spectrogram import spectrogram
from ..frequency_ranges import freq_range
from ..visualization import chord_visualizer

def frequency_to_midi(frequency):
    midi_number = 69 + 12 * numpy.log2(frequency / 440.0)
    return midi_number

def note_detection(spectrum, bin_size, start_freq=0):
    note_strength = numpy.zeros(12)
    sheet = []
    for t in range(spectrum.shape[1]):
        threshold = 0.3 * spectrum[:,t].max()
        for f in range(spectrum.shape[0]):
            if spectrum[f,t] <= threshold:
                continue
            else:
                midi_number = frequency_to_midi((f * bin_size) + start_freq)
                note_strength[int(midi_number) % 12] += spectrum[f,t]
        sheet.append(note_strength)
        note_strength = numpy.zeros(12)

    #sheet = sheet / numpy.max(sheet)
    print(numpy.max(sheet))
    print(numpy.min(sheet))
    print(numpy.count_nonzero(sheet))
    sheet = numpy.array(sheet).T
    return sheet


def chord_analyzer():
    path = "music\\scale.mp3"

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
    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)
    sheet = note_detection(spectrum.full_range[int(1*frame_len/sample_rate):int(4000*frame_len/sample_rate)], bin_size, start_freq=1)
    chord_visualizer(sheet, "ode_to_joy")
    print(sheet[5][150])

chord_analyzer()
