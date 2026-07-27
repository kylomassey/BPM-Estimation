import librosa
import numpy

from src.components.spectrogram import spectrogram


def _frame(y, frame_len, hop_len):
    return librosa.util.frame(x=y, frame_length=frame_len, hop_length=hop_len)


def test_spectrogram_shape_matches_frame_count_and_bins(tone_factory):
    sample_rate = 22050
    frame_len = 1024
    hop_len = 256
    y = tone_factory(freq=440.0, sample_rate=sample_rate, duration_s=1.0)
    framed = _frame(y, frame_len, hop_len)

    spectrum = spectrogram(framed)

    n_expected_bins = frame_len // 2 + 1
    assert spectrum.shape == (n_expected_bins, framed.shape[1])


def test_spectrogram_is_non_negative_power():
    # Power spectrum (|FFT|^2) can never be negative.
    rng = numpy.random.default_rng(0)
    framed = rng.standard_normal((512, 10))
    spectrum = spectrogram(framed)
    assert numpy.all(spectrum >= 0)


def test_spectrogram_silence_is_near_zero():
    framed = numpy.zeros((512, 5))
    spectrum = spectrogram(framed)
    assert numpy.allclose(spectrum, 0.0)


def test_spectrogram_detects_known_tone_frequency(tone_factory):
    sample_rate = 22050
    frame_len = 2048
    hop_len = 512
    freq = 440.0
    y = tone_factory(freq=freq, sample_rate=sample_rate, duration_s=1.0)
    framed = _frame(y, frame_len, hop_len)

    spectrum = spectrogram(framed)
    avg_spectrum = spectrum.mean(axis=1)
    bin_size = sample_rate / frame_len
    peak_bin = numpy.argmax(avg_spectrum)
    peak_freq = peak_bin * bin_size

    # FFT bin resolution here is ~10.8 Hz; a pure 440 Hz tone's energy
    # should land within one bin of the true frequency.
    assert abs(peak_freq - freq) <= bin_size
