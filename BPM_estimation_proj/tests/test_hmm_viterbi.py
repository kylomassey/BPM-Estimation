import numpy

from src.components.note_detection.hmm_viterbi import scores_to_emissions, viterbi


def test_scores_to_emissions_columns_sum_to_one():
    rng = numpy.random.default_rng(3)
    scores = rng.random((24, 10))
    emissions = scores_to_emissions(scores)
    assert numpy.allclose(emissions.sum(axis=0), 1.0)


def test_scores_to_emissions_clips_negative_scores():
    scores = numpy.array([[-1.0, 2.0], [3.0, -5.0]])
    emissions = scores_to_emissions(scores)
    assert numpy.all(emissions >= 0)


def test_real_transition_matrix_rows_are_valid_probability_distributions(project_root_cwd):
    a = numpy.load("transmission/transition_matrix.npy")
    assert a.shape == (24, 24)
    assert numpy.all(a >= 0)
    assert numpy.allclose(a.sum(axis=1), 1.0)
    # README claims a strong self-transition prior (chords tend to hold).
    assert numpy.all(numpy.diag(a) > 0.5)


def test_viterbi_smooths_a_single_frame_outlier(monkeypatch):
    # viterbi() hardcodes 24 states internally (one per major/minor chord),
    # so the transition matrix fixture must also be 24x24 regardless of how
    # many states the test actually exercises.
    n_states = 24
    off_diag = 0.05 / (n_states - 1)
    transition = numpy.full((n_states, n_states), off_diag)
    numpy.fill_diagonal(transition, 0.85)

    monkeypatch.setattr(
        "src.components.note_detection.hmm_viterbi.numpy.load",
        lambda path: transition,
    )

    n_frames = 9
    emissions = numpy.full((n_states, n_frames), 0.01)
    emissions[2, :] = 0.97  # true chord is state 2 throughout
    emissions[:, 4] = 0.01
    emissions[1, 4] = 0.97  # one noisy frame briefly "argmax"-favors state 1

    raw_argmax = numpy.argmax(emissions, axis=0)
    assert raw_argmax[4] == 1  # confirms the outlier really would fool a frame-wise decoder

    path = viterbi(emissions)

    # Viterbi, backed by a self-transition-favoring prior, should not
    # get pulled off course by one noisy frame.
    assert numpy.all(path == 2)


def test_viterbi_returns_valid_state_path_length(monkeypatch):
    n_states = 24
    transition = numpy.full((n_states, n_states), 1.0 / n_states)
    monkeypatch.setattr(
        "src.components.note_detection.hmm_viterbi.numpy.load",
        lambda path: transition,
    )
    emissions = numpy.full((n_states, 6), 1.0 / n_states)
    path = viterbi(emissions)
    assert path.shape == (6,)
    assert numpy.all((path >= 0) & (path < n_states))
