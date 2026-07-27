"""Smoke tests for the interactive top-level pipeline functions.

These call the same functions main.py wires up to the CLI menu. They only
exercise the missing-file guard clause, which is the one code path in
bpm_estimation()/stft_chord_analyzer()/cqt_chord_analyzer() that returns
before hitting input() or writing plots to charts/ -- so it can run safely
and silently under pytest without any mocking.
"""

from src.components.bpm_estimation import bpm_estimation
from src.components.note_detection.note_detection import (
    stft_chord_analyzer,
    cqt_chord_analyzer,
)


def test_bpm_estimation_missing_file_is_handled_gracefully(capsys):
    result = bpm_estimation("music/does_not_exist_at_all.wav", "does_not_exist_at_all.wav")
    assert result is True
    assert "File not found" in capsys.readouterr().out


def test_stft_chord_analyzer_missing_file_is_handled_gracefully(capsys):
    result = stft_chord_analyzer("music/does_not_exist_at_all.wav", "does_not_exist_at_all.wav")
    assert result is True
    assert "File not found" in capsys.readouterr().out


def test_cqt_chord_analyzer_missing_file_is_handled_gracefully(capsys):
    result = cqt_chord_analyzer("music/does_not_exist_at_all.wav", "does_not_exist_at_all.wav")
    assert result is True
    assert "File not found" in capsys.readouterr().out
