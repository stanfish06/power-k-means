import numpy as np
from numpy.typing import NDArray

def power_kmeans(
    X: NDArray[np.float64],
    k: int,
    s0: float,
    eta: float,
    atol: float,
    max_iters: int,
    seed: int,
    batch_size: int,
    alpha: float,
    verbose: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float]: ...
