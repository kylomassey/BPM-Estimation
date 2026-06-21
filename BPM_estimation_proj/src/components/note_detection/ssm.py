import numpy

def self_similarity_matrix (x):
    x = numpy.asarray(x, dtype=numpy.float32)
    norms = numpy.linalg.norm(x=x, axis=0, keepdims=True)
    x_norm = x/(norms + 1e-9)
    return x_norm.T @ x_norm