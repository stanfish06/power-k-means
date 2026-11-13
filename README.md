An implementation of power k-means ([Power k-Means Clustering](https://proceedings.mlr.press/v97/xu19a.html)) using c++.

## Python

```bash
pip install power-k-means
```

```python
import numpy as np
from power_kmeans import PowerKMeans, power_kmeans

X = np.random.default_rng(0).normal(size=(1000, 2))

res = power_kmeans(X, 4, s0=-1.0, eta=1.05, tol=1e-6, max_iter=100, random_state=0)
res.centroids, res.labels, res.membership, res.converged, res.n_iter, res.power

model = PowerKMeans(n_clusters=4, random_state=0).fit(X)
model.cluster_centers_, model.labels_, model.predict(X)
```

`batch_size` and `alpha` turn on mini-batch updates with moving averages of the
weighted sums. `verbose=1` prints progress and `verbose=2` also prints matrices;
both go to the process stdout from C++.

The power grows by a factor `eta` per update. Past roughly `-100` (depending on
the smallest point-to-centroid distance) `pow(distance, power)` overflows and
the centroids become NaN; the Python helper warns when that happens. Runs that
do not reach `tol` (mini-batch runs in particular) need `max_iter` or `eta`
small enough to stay below that point.
