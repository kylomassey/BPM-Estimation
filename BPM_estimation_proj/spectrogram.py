#Spectrogram.py
import matplotlib.pyplot as plt
import numpy

def displaySpectrogram(spectrum, hop_time):
    spectrum_db =  20 * numpy.log10(spectrum / numpy.max(spectrum) + 1e-10)
    time_scale = numpy.arange(spectrum.shape[1]) * hop_time
    freq_scale = numpy.arange(spectrum.shape[0]) * 20
    spectrum_db = numpy.maximum(spectrum_db, -100)
    plt.imshow(
        spectrum_db,
        origin = "lower",
        aspect = "auto",
        cmap = "magma",
        extent = [time_scale[0], time_scale[-1], freq_scale[0], freq_scale[-1]]
        )
    plt.colorbar(label='dB')
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.savefig("Spectrogram.png", dpi = 300, bbox_inches = "tight")

def spectrogram(audio, hop_time):
    spectrum = []
    window = numpy.hanning(audio.shape[0])
    
    for f in range(len(audio[0])):
        windowed_audio_frame = audio[:,f] * window
        fft_audio_frame =  numpy.fft.rfft(windowed_audio_frame)
        spectrum.append(numpy.abs(fft_audio_frame) ** 2)
    spectrum = numpy.array(spectrum).T
    displaySpectrogram(spectrum, hop_time)
    return(spectrum)