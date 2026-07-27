import numpy
import pytest

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root_cwd(monkeypatch):
    """Some modules (e.g. hmm_viterbi) load files via paths relative to the
    project root (``transmission/transition_matrix.npy``). Run inside that
    directory so those relative paths resolve the way they do when the app
    is launched via ``python -m src.main``."""
    monkeypatch.chdir(PROJECT_ROOT)


def make_tone(freq, sample_rate, duration_s, amplitude=1.0):
    """Pure sine tone of length duration_s at sample_rate, in samples."""
    n = int(sample_rate * duration_s)
    t = numpy.arange(n) / sample_rate
    return (amplitude * numpy.sin(2 * numpy.pi * freq * t)).astype(numpy.float64)


def make_click_track(bpm, sample_rate, duration_s, click_freq=2000.0, click_len_s=0.01):
    """Synthetic percussive click track with a known, exact tempo.

    A short decaying tone burst is placed at every beat; everything else is
    silence. This gives a clean, known-ground-truth signal for testing the
    onset/autocorrelation/tempogram pipeline, the same idea the project's
    own README proposes ("known-BPM click tracks") for validation.
    """
    n_total = int(sample_rate * duration_s)
    y = numpy.zeros(n_total, dtype=numpy.float64)
    beat_period_s = 60.0 / bpm
    click_len_n = int(sample_rate * click_len_s)
    envelope = numpy.exp(-numpy.linspace(0, 12, click_len_n))
    t_click = numpy.arange(click_len_n) / sample_rate
    click = envelope * numpy.sin(2 * numpy.pi * click_freq * t_click)

    beat_time = 0.0
    while beat_time < duration_s:
        start = int(beat_time * sample_rate)
        end = min(start + click_len_n, n_total)
        y[start:end] += click[: end - start]
        beat_time += beat_period_s
    return y


@pytest.fixture
def tone_factory():
    return make_tone


@pytest.fixture
def click_track_factory():
    return make_click_track
