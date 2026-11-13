#pragma once

#include <boost/numeric/ublas/matrix.hpp>

namespace pkm {

struct Result {
  // n_obs x k soft assignments from each observation's last visit (rows of
  // observations never visited in mini-batch mode stay zero)
  boost::numeric::ublas::matrix<double> membership;
  // k x n_feats cluster centres
  boost::numeric::ublas::matrix<double> centroids;
  bool converged;
  // number of centroid updates performed
  int n_iter;
  // power s at exit
  double power;
};

// Power k-means (Xu & Lange, ICML 2019). X is n_obs x n_feats, row-major.
// The power starts at s0 and is multiplied by eta after every centroid update;
// the run stops when the summed absolute centroid change drops below atol or
// after max_iters updates.
// batch_size: observations per update; 0 or >= n_obs uses all of them.
// alpha: weight of each new batch in the moving averages of the weighted sums;
//   1 keeps only the current batch.
// verbose: 0 silent, 1 per-iteration progress lines, 2 also dumps matrices.
// Throws std::invalid_argument on invalid sizes or parameters.
Result power_kmeans(boost::numeric::ublas::matrix<double> X, int k, double s0,
                    double eta, double atol, int max_iters, int seed,
                    int batch_size = 0, double alpha = 1.0, int verbose = 0);

} // namespace pkm
