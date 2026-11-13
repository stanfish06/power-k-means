"""Test the Python API"""

from __future__ import annotations

import numpy as np
import pytest

from power_kmeans import PowerKMeans, PowerKMeansResult, power_kmeans

N_PER_CLUSTER = 150
TRUE_CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=np.float64)
NOISE_SD = 0.5
CENTER_ATOL = 0.2  # allowed distance between a fitted and a true centre


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(len(TRUE_CENTERS)), N_PER_CLUSTER)
    X = TRUE_CENTERS[labels] + rng.normal(0.0, NOISE_SD, size=(labels.size, 2))
    perm = rng.permutation(labels.size)
    return X[perm], labels[perm]


def _assert_recovers(X: np.ndarray, true_labels: np.ndarray, res: PowerKMeansResult) -> None:
    k = len(TRUE_CENTERS)
    assert res.centroids.shape == (k, 2)
    # every true centre has exactly one fitted centre nearby
    dist = np.linalg.norm(res.centroids[:, None, :] - TRUE_CENTERS[None, :, :], axis=2)
    match = dist.argmin(axis=0)
    assert sorted(match.tolist()) == list(range(k))
    assert dist[match, np.arange(k)].max() < CENTER_ATOL
    # labels are a relabelling of the truth
    assert np.array_equal(match[true_labels], res.labels)


def test_recovers_separated_blobs(blobs) -> None:
    X, y = blobs
    res = power_kmeans(X, 4, random_state=1)
    assert res.converged
    assert 0 < res.n_iter <= 100
    assert res.membership.shape == (X.shape[0], 4)
    np.testing.assert_allclose(res.membership.sum(axis=1), 1.0, rtol=1e-12)
    _assert_recovers(X, y, res)


def test_minibatch_recovers_separated_blobs(blobs) -> None:
    X, y = blobs
    res = power_kmeans(X, 4, random_state=1, batch_size=100, alpha=0.3, tol=0.0, max_iter=60)
    _assert_recovers(X, y, res)


def test_estimator_matches_function(blobs) -> None:
    X, _ = blobs
    model = PowerKMeans(n_clusters=4, random_state=3).fit(X)
    res = power_kmeans(X, 4, random_state=3)
    assert np.array_equal(model.cluster_centers_, res.centroids)
    assert np.array_equal(model.labels_, res.labels)
    assert np.array_equal(model.predict(X), model.labels_)
    assert np.array_equal(PowerKMeans(n_clusters=4, random_state=3).fit_predict(X), res.labels)
    assert model.n_iter_ == res.n_iter and model.converged_ == res.converged


def test_same_seed_same_result(blobs) -> None:
    X, _ = blobs
    # max_iter=40 keeps the power above about -7, far from pow() overflow
    kwargs = {"random_state": 5, "batch_size": 64, "alpha": 0.5, "tol": 0.0, "max_iter": 40}
    a = power_kmeans(X, 4, **kwargs)
    b = power_kmeans(X, 4, **kwargs)
    assert np.isfinite(a.centroids).all()
    assert np.array_equal(a.centroids, b.centroids)
    assert np.array_equal(a.membership, b.membership)


def test_input_layouts_give_same_result(blobs) -> None:
    X, _ = blobs
    base = power_kmeans(X, 4).centroids
    assert np.array_equal(power_kmeans(np.asfortranarray(X), 4).centroids, base)
    assert np.array_equal(power_kmeans(X.tolist(), 4).centroids, base)
    # read-only input is accepted
    ro = X.copy()
    ro.flags.writeable = False
    assert np.array_equal(power_kmeans(ro, 4).centroids, base)


def test_verbose_zero_is_silent(blobs, capfd) -> None:
    X, _ = blobs
    power_kmeans(X, 4, verbose=0)
    assert capfd.readouterr().out == ""
    power_kmeans(X, 4, verbose=1, max_iter=2, tol=0.0)
    out = capfd.readouterr().out
    assert "[INFO] Start iteration 0" in out and "[DEBUG]" not in out


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_clusters": 0}, "k must be"),
        ({"n_clusters": 1000}, "k must be"),
        ({"s0": 0.0}, "s0"),
        ({"eta": -1.0}, "eta"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"max_iter": -1}, "max_iters"),
        ({"random_state": -1}, "random_state"),
    ],
)
def test_invalid_arguments(blobs, kwargs, match) -> None:
    X, _ = blobs
    args = {"n_clusters": 4, **kwargs}
    with pytest.raises(ValueError, match=match):
        power_kmeans(X, args.pop("n_clusters"), **args)


@pytest.mark.parametrize(
    "X",
    [np.zeros(5), np.array([[1.0, np.nan], [0.0, 1.0]]), np.empty((0, 2))],
    ids=["1d", "nan", "empty"],
)
def test_invalid_input(X) -> None:
    with pytest.raises(ValueError):
        power_kmeans(X, 1)


def test_predict_requires_fit_and_matching_features(blobs) -> None:
    X, _ = blobs
    with pytest.raises(RuntimeError):
        PowerKMeans(4).predict(X)
    model = PowerKMeans(4).fit(X)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((3, 5)))
