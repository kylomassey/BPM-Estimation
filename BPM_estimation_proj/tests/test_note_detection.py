import numpy

from src.components.note_detection.detection_tools import (
    frequency_to_midi,
    note_detection,
    chord_templates,
    match_chord,
)


def test_frequency_to_midi_known_reference_pitches():
    assert numpy.isclose(frequency_to_midi(440.0), 69.0)
    assert numpy.isclose(frequency_to_midi(880.0), 81.0)  # one octave up = +12
    assert numpy.isclose(frequency_to_midi(220.0), 57.0)  # one octave down = -12


def test_note_detection_folds_bin_energy_into_correct_pitch_class():
    # A single bin whose frequency lands exactly on 440 Hz (A, midi 69 ->
    # pitch class 9). start_freq is offset away from 0 Hz since
    # frequency_to_midi(0) is undefined (log2 of 0).
    spectrum = numpy.array([[5.0]])
    sheet = note_detection(spectrum, bin_size=440.0, start_freq=440.0)

    assert sheet.shape == (12, 1)
    a_pitch_class = 69 % 12
    assert sheet[a_pitch_class, 0] == 5.0
    assert sheet.sum() == 5.0  # no energy leaked into other pitch classes


def test_chord_templates_has_24_labeled_triads():
    templates, labels = chord_templates()
    assert templates.shape == (24, 12)
    assert len(labels) == 24
    assert labels[0] == "C major"
    assert labels[1] == "C minor"
    assert labels[2] == "C# major"


def test_chord_templates_root_note_bit_is_set():
    templates, labels = chord_templates()
    for i, label in enumerate(labels):
        root_index = i // 2  # majors/minors alternate, 2 labels per root note
        assert templates[i, root_index] == 1


def test_match_chord_identifies_exact_template_as_best_match():
    templates, labels = chord_templates()
    c_major_index = labels.index("C major")
    chroma = templates[c_major_index].astype(float).reshape(12, 1)

    scores, max_scores, best_labels, all_labels = match_chord(chroma)

    assert best_labels[0] == "C major"
    # The winning template should score a perfect (or near-perfect) match
    # against its own exact template.
    assert scores[c_major_index, 0] == numpy.max(scores[:, 0])
    assert numpy.isclose(scores[c_major_index, 0], 1.0, atol=1e-6)


def test_match_chord_max_scores_only_keeps_the_winner_per_frame():
    templates, labels = chord_templates()
    chroma = templates[0].astype(float).reshape(12, 1)
    _, max_scores, _, _ = match_chord(chroma)
    # Exactly one template should be non-zero in the "winner-take-all" output.
    assert numpy.count_nonzero(max_scores[:, 0]) == 1
