import numpy

def frequency_to_midi(frequency):
    midi_number = 69 + 12 * numpy.log2(frequency / 440.0)
    return midi_number

def note_detection(spectrum, bin_size, start_freq=0):
    n_freq = spectrum.shape[0]

    frequencies = numpy.arange(n_freq) * bin_size + start_freq
    midi = frequency_to_midi(frequencies)
    pitch_classes = numpy.round(midi).astype(int) % 12

    M = numpy.zeros((12,n_freq), dtype=spectrum.dtype)
    M[pitch_classes, numpy.arange(n_freq)] = 1.0
    sheet = M @ spectrum

    print(numpy.max(sheet))
    print(numpy.min(sheet))
    print(numpy.count_nonzero(sheet))
    return sheet

def chord_templates():
    major = numpy.array([1,0,0,0,1,0,0,1,0,0,0,0])
    minor = numpy.array([1,0,0,1,0,0,0,1,0,0,0,0])
    #dom7  = numpy.array([1,0,0,0,1,0,0,1,0,0,1,0])
    #min7  = numpy.array([1,0,0,1,0,0,0,1,0,0,1,0])
    #maj7  = numpy.array([1,0,0,0,1,0,0,1,0,0,0,1])
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    template, label = [], []
    for i in range(0,12):
        template.append(numpy.roll(major,i)); label.append(f"{notes[i]} major")
        template.append(numpy.roll(minor,i)); label.append(f"{notes[i]} minor")
        #template.append(numpy.roll(dom7,i)); label.append(f"{notes[i]} dom7")
        #template.append(numpy.roll(min7,i)); label.append(f"{notes[i]} min7")
        #template.append(numpy.roll(maj7,i)); label.append(f"{notes[i]} maj7")
    return numpy.array(template), label

def match_chord(chroma):
    template, label = chord_templates()
    t_norm = template / numpy.linalg.norm(template,axis=1,keepdims=True)
    c_norm = chroma / (numpy.linalg.norm(chroma, axis=0) + 1e-9)
    scores = t_norm @ c_norm
    col_max = scores.max(axis=0, keepdims=True)
    max_scores = numpy.where(scores == col_max, scores, 0)
    return scores, numpy.array(max_scores), numpy.array([label[i] for i in numpy.argmax(scores, axis=0)]), label
