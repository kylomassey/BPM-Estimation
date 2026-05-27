import matplotlib.pyplot as plt
import numpy

def display_spectrogram(spectrum, hop_time):
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
    plt.savefig("BPM_estimation_proj/charts/Spectrogram.png", dpi = 300, bbox_inches = "tight")