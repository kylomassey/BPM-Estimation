import matplotlib.pyplot as plt
import numpy

def display_spectrogram(spectrum, hop_time, filename, bin_size):
    spectrum_db =  20 * numpy.log10(spectrum / numpy.max(spectrum) + 1e-10)
    time_scale = numpy.arange(spectrum.shape[1]) * hop_time
    freq_scale = numpy.arange(spectrum.shape[0]) * bin_size
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
    plt.savefig(f"charts\\{filename}_Spectrogram.png", dpi = 300, bbox_inches = "tight")
    plt.clf()

def display_tempogram(tempogram, bpm_scale, filename):
    n_lags = tempogram.shape[0]
    time_scale = numpy.arange(tempogram.shape[1])
    plt.imshow(
        tempogram,
        origin = "lower",
        aspect = "auto",
        extent = [time_scale[0], time_scale[-1], 0, n_lags]
        )
    
    tick_positions = numpy.linspace(0, n_lags - 1, 8).astype(int)
    tick_labels = [f"{bpm_scale[i]:.0f}" for i in tick_positions]
    plt.yticks(tick_positions, tick_labels)

    plt.colorbar(label='tempo strength')
    plt.xlabel("Time frames")
    plt.ylabel("lag bins")
    plt.savefig(f"charts\\{filename}_Tempogram.png", dpi = 300, bbox_inches = "tight")
    plt.clf()

def chord_visualizer(sheet, filename):
    sheet = sheet / (sheet.max() + 1e-10)
    sheet = numpy.maximum(sheet, -100)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.imshow(
        sheet,
        origin = "lower",
        aspect = "auto",
        cmap = "magma",
        extent = [0, sheet.shape[1], 0, 12]
        )
    plt.colorbar(label='note strength')
    plt.yticks(numpy.arange(12), notes)
    plt.xlabel("Time frames")
    plt.ylabel("Pitch classes")
    plt.savefig(f"charts\\{filename}_Chord_Visualization.png", dpi = 300, bbox_inches = "tight")
    plt.clf()