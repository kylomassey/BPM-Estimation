import numpy

from src.components.novelty_curve import onset_curve, smooth_curve, process_band


def test_onset_curve_is_positive_diff_scaled_by_hop_time():
    # Column energy strictly increasing by 1 each frame.
    spectrum = numpy.tile(numpy.arange(5.0), (3, 1))  # shape (3 bins, 5 frames)
    hop_time = 0.5
    result = onset_curve(spectrum, hop_time)
    col_energy = spectrum.sum(axis=0)  # [0, 3, 6, 9, 12]
    expected = numpy.diff(col_energy) / hop_time
    assert numpy.allclose(result, expected)
    assert numpy.all(result >= 0)


def test_onset_curve_clips_negative_changes_to_zero():
    # Energy decreases every frame -> raw diff is negative everywhere.
    spectrum = numpy.tile(numpy.arange(5.0, 0, -1), (2, 1))
    result = onset_curve(spectrum, hop_time=1.0)
    assert numpy.all(result == 0)


def test_smooth_curve_preserves_length():
    onset = numpy.array([0.0, 1.0, 0.0, 1.0, 0.0])
    smoothed = smooth_curve(onset, framel=3)
    assert smoothed.shape == onset.shape


def test_smooth_curve_is_moving_sum_of_window():
    onset = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0])
    smoothed = smooth_curve(onset, framel=3)
    expected = numpy.convolve(onset, numpy.ones(3), mode="same")
    assert numpy.allclose(smoothed, expected)


def test_process_band_matches_manual_pipeline():
    spectrum = numpy.tile(numpy.array([0.0, 2.0, 2.0, 4.0]), (2, 1))
    hop = 0.25
    result = process_band(spectrum, hop)
    expected = smooth_curve(onset_curve(spectrum, hop), framel=max(1, int(0.1 / hop)))
    assert numpy.allclose(result, expected)
