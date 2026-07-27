import librosa
import numpy
import pytest

from src.components.estimation import auto_correlation, harmonic_scoring, tempogram
from src.components.spectrogram import spectrogram
from src.components.frequency_ranges import freq_range
from src.components.novelty_curve import process_band


def test_auto_correlation_peaks_at_true_period():
    # A clean periodic impulse train with period 8 samples.
    period = 8
    onset = numpy.tile([1.0, 0, 0, 0, 0, 0, 0, 0], 10)
    corr = auto_correlation(onset, laglow=2, laghigh=20)
    best_lag = numpy.argmax(corr) + 2
    assert best_lag == period


def test_auto_correlation_is_bounded():
    rng = numpy.random.default_rng(1)
    onset = rng.random(200)
    corr = auto_correlation(onset, laglow=1, laghigh=50)
    # Cosine-similarity-style normalization keeps correlation in [-1, 1]
    # (plus a tiny epsilon slack for near-zero-energy edge cases).
    assert numpy.all(numpy.array(corr) <= 1.0 + 1e-6)
    assert numpy.all(numpy.array(corr) >= -1.0 - 1e-6)


def test_harmonic_scoring_prefers_fundamental_over_octave_alias():
    # This is exactly the octave-error scenario the project's harmonic
    # scoring is designed to correct: a click train's autocorrelation has
    # comparably strong peaks at the true period AND at its multiples
    # (2x, 3x, 4x the period). harmonic_scoring should still land on the
    # smallest lag (the true fundamental), not one of its harmonics.
    laglow, laghigh = 2, 40
    true_period = 8
    corr = numpy.zeros(laghigh - laglow + 1)
    for k in range(true_period, laghigh + 1, true_period):
        corr[k - laglow] = 1.0
    best_idx = harmonic_scoring(corr, laglow, laghigh)
    best_lag = best_idx + laglow
    assert best_lag == true_period


def test_harmonic_scoring_does_not_crash_at_range_edges():
    laglow, laghigh = 2, 5
    corr = numpy.ones(laghigh - laglow + 1)
    idx = harmonic_scoring(corr, laglow, laghigh)
    assert laglow <= idx + laglow <= laghigh


def test_tempogram_shape():
    rng = numpy.random.default_rng(2)
    onset = rng.random(2000)
    hop_time = 0.0125  # 12.5ms hop, matches bpm_estimation's real hop size
    laglow, laghigh = 5, 40
    tgram, highscore = tempogram(onset, hop_time, laglow, laghigh)
    assert tgram.shape[0] == laghigh - laglow + 1
    assert 0 <= highscore < laghigh - laglow + 1


def _estimate_bpm_from_click_track(y, sample_rate):
    """Mirrors the numeric core of bpm_estimation() minus all I/O
    (no plotting, no interactive prompts) so it can run as a fast, silent
    pytest assertion."""
    frame_len = int(sample_rate * 0.05)
    hop_len = int(frame_len * 0.25)
    framed_audio = librosa.util.frame(x=y, frame_length=frame_len, hop_length=hop_len)
    hop_time = hop_len / sample_rate

    spectrum = spectrogram(framed_audio)
    spectrum = freq_range(spectrum, frame_len, sample_rate, hop_len)
    master_curve = process_band(spectrum.full_range, hop_time)

    bpm_low, bpm_high = 60, 220
    lag_low = int(60 / hop_time / bpm_high)
    lag_high = int(60 / hop_time / bpm_low)
    bpm_graph = numpy.divide(60, numpy.multiply(numpy.arange(lag_low, lag_high + 1), hop_time))

    corr = auto_correlation(master_curve, lag_low, lag_high)
    harmonic_bpm = bpm_graph[harmonic_scoring(corr, lag_low, lag_high)]
    return harmonic_bpm


def test_bpm_pipeline_recovers_known_click_track_tempo(click_track_factory):
    sample_rate = 22050
    true_bpm = 120.0
    y = click_track_factory(bpm=true_bpm, sample_rate=sample_rate, duration_s=15.0)

    estimated_bpm = _estimate_bpm_from_click_track(y, sample_rate)

    assert abs(estimated_bpm - true_bpm) <= 2.0


def test_bpm_pipeline_recovers_known_click_track_tempo_at_different_tempo(click_track_factory):
    sample_rate = 22050
    true_bpm = 95.0
    y = click_track_factory(bpm=true_bpm, sample_rate=sample_rate, duration_s=15.0)

    estimated_bpm = _estimate_bpm_from_click_track(y, sample_rate)

    assert abs(estimated_bpm - true_bpm) <= 2.0


@pytest.mark.parametrize("true_bpm", [128, 136, 140, 148, 156, 164, 172, 176])
def test_bpm_pipeline_does_not_lock_onto_octave_alias(click_track_factory, true_bpm):
    # Regression test for a real bug: harmonic_scoring's k // 2 backward
    # lookup let a candidate steal credit from its own half-lag (double
    # tempo) neighbor "for free," while the true fundamental only earned
    # credit by reaching its own 2x/3x/4x multiples -- which get cut off
    # by laghigh well before the alias's backward lookup does. That made
    # roughly the 128-176 BPM range consistently lock onto exactly half
    # the true tempo. Confirms it stays fixed.
    sample_rate = 22050
    y = click_track_factory(bpm=true_bpm, sample_rate=sample_rate, duration_s=15.0)

    estimated_bpm = _estimate_bpm_from_click_track(y, sample_rate)

    assert abs(estimated_bpm - true_bpm) <= 3.0
    assert not (0.4 < estimated_bpm / true_bpm < 0.6)  # would indicate an octave-down lock
