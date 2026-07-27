import numpy

from src.components.note_detection.adjustments import (
    median_smoothing,
    diagonal_smoothing,
    downsample_time,
)


def test_median_smoothing_removes_a_single_frame_spike():
    chroma = numpy.zeros((12, 21))
    chroma[3, :] = 1.0
    chroma[3, 10] = 100.0  # one-frame outlier
    smoothed = median_smoothing(chroma, filt_len=5)
    assert smoothed[3, 10] == 1.0


def test_diagonal_smoothing_with_filt_len_one_is_identity():
    rng = numpy.random.default_rng(4)
    ssm = rng.random((6, 6))
    result = diagonal_smoothing(ssm, filt_len=1)
    assert numpy.allclose(result, ssm)


def test_diagonal_smoothing_interior_of_constant_matrix_is_unchanged():
    filt_len = 3
    n = 10
    ssm = numpy.full((n, n), 5.0)
    result = diagonal_smoothing(ssm, filt_len=filt_len)
    # Away from the zero-padded edge, averaging a constant matrix along the
    # diagonal should still yield that same constant.
    interior = result[: n - filt_len, : n - filt_len]
    assert numpy.allclose(interior, 5.0)


def test_downsample_time_averages_blocks_of_known_values():
    chroma = numpy.zeros((12, 6))
    chroma[0] = [1, 1, 3, 3, 5, 5]
    result = downsample_time(chroma, factor=2)
    assert result.shape == (12, 3)
    assert numpy.allclose(result[0], [1, 3, 5])


def test_downsample_time_drops_remainder_frames():
    chroma = numpy.ones((12, 7))  # 7 not divisible by factor=2 -> last frame dropped
    result = downsample_time(chroma, factor=2)
    assert result.shape == (12, 3)
