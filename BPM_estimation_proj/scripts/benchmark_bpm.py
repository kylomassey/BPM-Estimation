"""Baseline comparison: this project's BPM estimator vs. librosa.beat.beat_track.

The README positions the hand-rolled autocorrelation/harmonic-scoring
estimator in bpm_estimation.py as an alternative to library beat trackers,
but the project had no quantitative evidence of how well it actually
performs. This script closes that gap using synthetic click tracks with
exactly known ground-truth tempo (no copyrighted audio required, so it's
safe to run and share results from).

Usage:
    python scripts/benchmark_bpm.py
"""
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.spectrogram import spectrogram
from src.components.frequency_ranges import freq_range
from src.components.novelty_curve import process_band
from src.components.estimation import auto_correlation, harmonic_scoring

SAMPLE_RATE = 22050
TEST_BPMS = [70, 85, 100, 115, 128, 140, 160, 175]
DURATION_S = 20.0


def make_click_track(bpm, sample_rate=SAMPLE_RATE, duration_s=DURATION_S,
                      click_freq=2000.0, click_len_s=0.01):
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


def estimate_project_bpm(y, sample_rate=SAMPLE_RATE):
    """Same numeric pipeline as src/components/bpm_estimation.py, with all
    plotting/interactive I/O stripped out."""
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
    return bpm_graph[harmonic_scoring(corr, lag_low, lag_high)]


def estimate_librosa_bpm(y, sample_rate=SAMPLE_RATE):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
    return float(numpy.asarray(tempo).ravel()[0])


def main():
    rows = []
    for true_bpm in TEST_BPMS:
        y = make_click_track(true_bpm)
        project_bpm = estimate_project_bpm(y)
        librosa_bpm = estimate_librosa_bpm(y)
        rows.append((true_bpm, project_bpm, librosa_bpm))

    print(f"{'true BPM':>10} {'this project':>14} {'librosa':>10} "
          f"{'project err':>13} {'librosa err':>13}")
    project_errors, librosa_errors = [], []
    for true_bpm, project_bpm, librosa_bpm in rows:
        project_err = abs(project_bpm - true_bpm)
        librosa_err = abs(librosa_bpm - true_bpm)
        project_errors.append(project_err)
        librosa_errors.append(librosa_err)
        print(f"{true_bpm:>10} {project_bpm:>14.1f} {librosa_bpm:>10.1f} "
              f"{project_err:>13.1f} {librosa_err:>13.1f}")

    print(f"\nMean absolute error -- this project: {numpy.mean(project_errors):.2f} BPM, "
          f"librosa: {numpy.mean(librosa_errors):.2f} BPM")

    charts_dir = PROJECT_ROOT / "charts"
    charts_dir.mkdir(exist_ok=True)

    x = numpy.arange(len(TEST_BPMS))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, project_errors, width, label="this project")
    plt.bar(x + width / 2, librosa_errors, width, label="librosa.beat.beat_track")
    plt.xticks(x, [str(b) for b in TEST_BPMS])
    plt.xlabel("True BPM (synthetic click track)")
    plt.ylabel("Absolute error (BPM)")
    plt.title("BPM estimation error vs. librosa baseline")
    plt.legend()
    out_path = charts_dir / "bpm_baseline_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison chart to {out_path}")


if __name__ == "__main__":
    main()
