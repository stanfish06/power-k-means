from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core

__all__ = ["PowerKMeans", "PowerKMeansResult", "power_kmeans"]

# Defaults shared by power_kmeans() and PowerKMeans.
DEFAULT_S0 = -1.0  # initial power; -1 is the harmonic mean
DEFAULT_ETA = 1.05  # power multiplier per update
DEFAULT_TOL = 1e-6  # stop when the summed |centroid change| drops below this
# With eta=1.05 the power passes about -130 near update 100; beyond that
# pow(d, s) can overflow and the result turns NaN.
DEFAULT_MAX_ITER = 100

_INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True)
class PowerKMeansResult:
    """Output of :func:`power_kmeans`.

    centroids: (n_clusters, n_features) cluster centres.
    membership: (n_samples, n_clusters) soft assignments from the last update
        each sample took part in; in mini-batch mode rows of samples never
        drawn are zero.
    labels: (n_samples,) index of the nearest centroid.
    converged: whether the tolerance was reached before max_iter.
    n_iter: number of centroid updates performed.
    power: power s at exit.
    """

    centroids: NDArray[np.float64]
    membership: NDArray[np.float64]
    labels: NDArray[np.intp]
    converged: bool
    n_iter: int
    power: float


def _as_matrix(X: ArrayLike, name: str = "X") -> NDArray[np.float64]:
    arr = np.ascontiguousarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _nearest(X: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.intp]:
    d2 = (
        np.einsum("ij,ij->i", X, X)[:, None]
        - 2.0 * X @ centroids.T
        + np.einsum("ij,ij->i", centroids, centroids)[None, :]
    )
    return np.argmin(d2, axis=1)


def _seed(random_state: int | None) -> int:
    if random_state is None:
        return int(np.random.default_rng().integers(0, _INT32_MAX))
    seed = int(random_state)
    if not 0 <= seed <= _INT32_MAX:
        raise ValueError(f"random_state must be in [0, {_INT32_MAX}], got {seed}")
    return seed


def power_kmeans(
    X: ArrayLike,
    n_clusters: int,
    *,
    s0: float = DEFAULT_S0,
    eta: float = DEFAULT_ETA,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    random_state: int | None = 0,
    batch_size: int | None = None,
    alpha: float = 1.0,
    verbose: int = 0,
) -> PowerKMeansResult:
    """Cluster the rows of X with power k-means

    Parameters
    ----------
    X : (n_samples, n_features)
    n_clusters : number of clusters, 1 <= n_clusters <= n_samples.
    s0 : initial power (non-zero); negative values start between the harmonic
        mean and the minimum.
    eta : factor applied to the power after every update; > 1 drives the power
        towards -inf, where the objective becomes the k-means objective.
    tol : stop when the summed absolute change of all centroids is below tol.
    max_iter : maximum number of centroid updates.
    random_state : seed for centroid initialisation and batch shuffling; None
        draws a fresh seed.
    batch_size : samples per update; None uses all samples.
    alpha : weight of each new batch in the moving averages, in (0, 1];
        1 keeps only the current batch.
    verbose : 0 silent, 1 progress lines, 2 also prints matrices. Output goes
        to the process stdout from C++, not to sys.stdout.
    """
    arr = _as_matrix(X)
    centroids, membership, converged, n_iter, power = _core.power_kmeans(
        arr,
        int(n_clusters),
        float(s0),
        float(eta),
        float(tol),
        int(max_iter),
        _seed(random_state),
        0 if batch_size is None else int(batch_size),
        float(alpha),
        int(verbose),
    )
    if not np.isfinite(centroids).all():
        warnings.warn(
            "power k-means produced non-finite centroids: pow(distance, power) "
            "overflowed or a distance was exactly zero. Lower max_iter or eta, "
            "or rescale X.",
            RuntimeWarning,
            stacklevel=2,
        )
        labels = np.full(arr.shape[0], -1, dtype=np.intp)
    else:
        labels = _nearest(arr, centroids)
    return PowerKMeansResult(
        centroids=centroids,
        membership=membership,
        labels=labels,
        converged=bool(converged),
        n_iter=int(n_iter),
        power=float(power),
    )


class PowerKMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        *,
        s0: float = DEFAULT_S0,
        eta: float = DEFAULT_ETA,
        tol: float = DEFAULT_TOL,
        max_iter: int = DEFAULT_MAX_ITER,
        random_state: int | None = 0,
        batch_size: int | None = None,
        alpha: float = 1.0,
        verbose: int = 0,
    ) -> None:
        self.n_clusters = n_clusters
        self.s0 = s0
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: object = None) -> PowerKMeans:
        res = power_kmeans(
            X,
            self.n_clusters,
            s0=self.s0,
            eta=self.eta,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            batch_size=self.batch_size,
            alpha=self.alpha,
            verbose=self.verbose,
        )
        self.cluster_centers_ = res.centroids
        self.labels_ = res.labels
        self.membership_ = res.membership
        self.converged_ = res.converged
        self.n_iter_ = res.n_iter
        self.power_ = res.power
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Index of the nearest fitted centroid for each row of X."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("PowerKMeans instance is not fitted; call fit first")
        arr = _as_matrix(X)
        if arr.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                f"X has {arr.shape[1]} features, the model was fitted with "
                f"{self.cluster_centers_.shape[1]}"
            )
        return _nearest(arr, self.cluster_centers_)

    def fit_predict(self, X: ArrayLike, y: object = None) -> NDArray[np.intp]:
        return self.fit(X).labels_
