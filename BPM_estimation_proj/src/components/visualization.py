import matplotlib.pyplot as plt
import numpy

def display_spectrogram(spectrum, hop_time, filename, bin_size = 1):
    spectrum_db =  20 * numpy.log10((spectrum / numpy.mean(spectrum)) + 1e-10)
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
    plt.savefig(f"charts/{filename}_Spectrogram.png", dpi = 300, bbox_inches = "tight")
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
    plt.savefig(f"charts/{filename}_Tempogram.png", dpi = 300, bbox_inches = "tight")
    plt.clf()

def display_chromagram(sheet, filename, hop_time):
    sheet = sheet/(sheet.mean(axis=0, keepdims=True) + 1e-9)
    time_scale = numpy.arange(sheet.shape[1]) * hop_time
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    plt.imshow(
        sheet,
        origin = "lower",
        aspect = "auto",
        cmap = "magma",
        extent = [time_scale[0], time_scale[-1], 0, 12]
        )
    plt.colorbar(label='note strength')
    plt.yticks(numpy.arange(12), notes)
    plt.xlabel("Time frames")
    plt.ylabel("Pitch classes")
    plt.savefig(f"charts/{filename}_Chromagram.png", dpi = 300, bbox_inches = "tight")
    plt.clf()

def display_ssm(sheet, hop_time, filename, factor = 1):
    #sheet = 20 * numpy.log10(sheet/numpy.max(sheet) + 1e-10)
    time_scale = numpy.arange(sheet.shape[1]) * hop_time * factor
    plt.imshow(
        sheet,
        origin = "lower",
        aspect = "auto",
        cmap = "Spectral",
        extent = [time_scale[0], time_scale[-1], time_scale[0], time_scale[-1]]
        )
    plt.colorbar(label='dB')
    plt.xlabel("Time frames")
    plt.ylabel("Time frames")
    plt.savefig(f"charts/{filename}_ssm.png", dpi = 300, bbox_inches = "tight")
    plt.clf()

def display_chords(sheet, labels, filename):
    print(sheet.shape)
    sheet = numpy.maximum(sheet, -100)
    print(labels.shape)
    chord_types= ["major", "minor", "dominant 7th", "minor 7th", "major 7th"]
    for i in range(0,2):
        plt.imshow(
            sheet[numpy.arange(0+i,sheet.shape[0],2),:],
            origin = "lower",
            aspect = "auto",
            cmap = "magma",
            extent = [0, sheet.shape[1], 0, 12]
            )
        plt.colorbar(label='note strength')
        plt.yticks(numpy.arange(12), labels[numpy.arange(0+i,labels.shape[0],2)])
        plt.xlabel("Time frames")
        plt.ylabel("Pitch classes")
        plt.savefig(f"charts/{filename}_{chord_types[i]}_Chord_Visualization.png", dpi = 300, bbox_inches = "tight")
        plt.clf()

def display_viterbi(emissions, path, hop_time, filename):
    names = [f"{n}{q}" for n in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                       for q in ['', 'm']]
    time_scale = numpy.arange(emissions.shape[1]) * hop_time

    plt.figure(figsize=(14, 6))
    plt.imshow(
        emissions,
        origin = "lower",
        aspect = "auto",
        cmap = "magma",
        extent = [time_scale[0], time_scale[-1], -0.5, 23.5]
        )
    raw = numpy.argmax(emissions, axis = 0)
    plt.plot(time_scale, raw, '.', color = "gray", markersize = 2, label = "frame-wise argmax")
    plt.plot(time_scale, path, color = "cyan", linewidth = 1.5,
             drawstyle = "steps-post", label = "viterbi")
    plt.colorbar(label = 'emission prob')
    plt.yticks(numpy.arange(24), names, fontsize = 6)
    plt.xlabel("Time (s)")
    plt.ylabel("Chord state")
    plt.legend(loc = "upper right")
    plt.savefig(f"charts/{filename}_Decode.png", dpi = 300, bbox_inches = "tight")
    plt.clf()