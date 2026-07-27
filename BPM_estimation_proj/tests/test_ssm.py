import numpy

from src.components.note_detection.ssm import self_similarity_matrix


def test_self_similarity_diagonal_is_one():
    rng = numpy.random.default_rng(5)
    chroma = rng.random((12, 8)) + 0.1  # avoid all-zero columns
    ssm = self_similarity_matrix(chroma)
    assert numpy.allclose(numpy.diag(ssm), 1.0, atol=1e-5)


def test_self_similarity_is_symmetric():
    rng = numpy.random.default_rng(6)
    chroma = rng.random((12, 8))
    ssm = self_similarity_matrix(chroma)
    assert numpy.allclose(ssm, ssm.T)


def test_self_similarity_identical_columns_are_perfectly_similar():
    chroma = numpy.tile(numpy.array([[1], [0], [0], [1]] + [[0]] * 8, dtype=float), (1, 3))
    ssm = self_similarity_matrix(chroma)
    assert numpy.allclose(ssm, 1.0, atol=1e-5)


def test_self_similarity_zero_column_does_not_produce_nan():
    chroma = numpy.zeros((12, 3))
    chroma[0, 1] = 1.0  # only one non-silent frame
    ssm = self_similarity_matrix(chroma)
    assert numpy.all(numpy.isfinite(ssm))
