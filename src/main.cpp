#include <algorithm>
#include <boost/format.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>

using namespace boost::numeric::ublas;
struct Result {
  matrix<double> membership;
  matrix<double> centroids;
  bool converged;
};

Result power_kmeans(matrix<double> X, int k, double s0, double eta, double atol,
                    int max_iters, int seed, int batch_size = 0,
                    double alpha = 1.0) {
  int n_obs = X.size1();
  int n_feats = X.size2();
  matrix<double> centroids(
      k, n_feats); // need initialization matrix<double> weights(n_obs, k);
  vector<int> idx(n_obs);
  std::iota(idx.begin(), idx.end(), 0);
  std::mt19937 rng(seed);
  for (int ki = 0; ki < k; ki++) {
    int oi = std::uniform_int_distribution<int>(ki, n_obs - 1)(rng);
    std::swap(idx[ki], idx[oi]);
  }
  for (int ki = 0; ki < k; ki++) {
    int oi = idx[ki];
    matrix_row<matrix<double>> obs_i(X, oi);
    matrix_row<matrix<double>> centroid_i(centroids, ki);
    centroid_i = obs_i;
    for (int j = 0; j < centroids.size2(); j++)
      centroid_i(j) += std::normal_distribution<double>(0.0, 1e-3)(rng);
  }
  // rows of observations outside the current batch keep the values from
  // their last visit, or zero if they were never visited
  matrix<double> weights(n_obs, k, 0.0);
  matrix<double> membership(n_obs, k, 0.0);
  matrix<double> batch_sums(k, n_feats);
  vector<double> weight_sums(k);
  // exponential moving averages of batch_sums and weight_sums
  matrix<double> ema_sums(k, n_feats, 0.0);
  vector<double> ema_mass(k, 0.0);
  // batches are consecutive slices of order; pos is the next unread entry
  if (batch_size <= 0 || batch_size > n_obs)
    batch_size = n_obs;
  vector<int> order(n_obs);
  std::iota(order.begin(), order.end(), 0);
  std::size_t pos = 0;
  // row-major storage of the matrices that are never reallocated
  const std::size_t nk = k, nf = n_feats;
  const double *x = &X.data()[0];
  const double *w = &weights.data()[0];
  double s = s0;
  int status = 0;

  auto start = std::chrono::high_resolution_clock::now();
  int i;
  for (i = 0; i < max_iters; i++) {
    std::cout << "[DEBUG] Centroids: " << centroids << std::endl;
    std::cout << boost::format("[INFO] Start iteration %1% with power %2%") %
                     i % s
              << std::endl;

    // Start a new pass once every observation has been read. Mini-batches
    // reshuffle the order for each pass; a full batch keeps the input order.
    if (pos == (std::size_t)n_obs)
      pos = 0;
    if (pos == 0 && batch_size < n_obs)
      std::shuffle(order.begin(), order.end(), rng);
    std::size_t batch_n = std::min<std::size_t>(batch_size, n_obs - pos);

    std::cout << "[INFO] Computing weights" << std::endl;
    auto start_wei = std::chrono::high_resolution_clock::now();
    for (std::size_t b = 0; b < batch_n; b++) {
      int oi = order[pos + b];
      vector<double> dists_i(k);
      vector<double> power_dists_i(k);
      for (int ki = 0; ki < k; ki++) {
        matrix_row<matrix<double>> obs_i(X, oi);
        matrix_row<matrix<double>> centroid_i(centroids, ki);
        dists_i[ki] = norm_2_square(obs_i - centroid_i);
        power_dists_i[ki] = pow(dists_i[ki], s);
      }
      double _a = pow(sum(power_dists_i), 1 / s - 1);
      vector<double> _b = element_div(power_dists_i, dists_i);
      matrix_row<matrix<double>> _row_weights(weights, oi);
      matrix_row<matrix<double>> _row_membership(membership, oi);
      _row_membership = _b / sum(_b);
      std::cout << "[DEBUG] _a: " << _a << std::endl;
      std::cout << "[DEBUG] _b: " << _b << std::endl;
      _b *= _a;
      std::cout << "[DEBUG] _b: " << _b << std::endl;
      _row_weights = _b;
    }
    std::cout << "[DEBUG] Weights: " << weights << std::endl;
    auto end_wei = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_wei - start_wei);
    std::cout << "[SUCCESS] Weights computation finished in: "
              << duration.count() << "ms" << std::endl;

    // Batch statistics: batch_sums = weights^T * X and weight_sums = column
    // sums of weights, over the rows of the batch. Each observation adds
    // w_ik * x_i to row k, so every inner loop runs over contiguous memory
    // instead of striding down a column of weights.
    double *bs = &batch_sums.data()[0];
    std::fill(bs, bs + nk * nf, 0.0);
    std::fill(weight_sums.begin(), weight_sums.end(), 0.0);
    for (std::size_t b = 0; b < batch_n; b++) {
      std::size_t oi = order[pos + b];
      const double *x_i = x + oi * nf;
      const double *w_i = w + oi * nk;
      for (std::size_t ki = 0; ki < nk; ki++) {
        double w_ik = w_i[ki];
        double *bs_k = bs + ki * nf;
        weight_sums[ki] += w_ik;
        for (std::size_t j = 0; j < nf; j++)
          bs_k[j] += w_ik * x_i[j];
      }
    }
    pos += batch_n;
    // Blend the batch statistics into the moving averages and set each
    // centroid to (average weighted sum) / (average weight). A batch from t
    // updates ago keeps the factor (1 - alpha)^t.
    double centroids_tot_chg_abs = 0;
    for (int ki = 0; ki < k; ki++) {
      matrix_row<matrix<double>> _batch_sum_i(batch_sums, ki);
      matrix_row<matrix<double>> _ema_sum_i(ema_sums, ki);
      _ema_sum_i = (1 - alpha) * _ema_sum_i + alpha * _batch_sum_i;
      ema_mass[ki] = (1 - alpha) * ema_mass[ki] + alpha * weight_sums[ki];
      vector<double> _new_centroid_i = _ema_sum_i / ema_mass[ki];
      matrix_row<matrix<double>> _old_centroid_i(centroids, ki);
      centroids_tot_chg_abs += norm_1(_new_centroid_i - _old_centroid_i);
      _old_centroid_i = _new_centroid_i;
    }
    std::cout << "[INFO] Absolute centroids change: " << centroids_tot_chg_abs
              << std::endl;
    if (centroids_tot_chg_abs < atol) {
      std::cout << "[INFO] Weights computation finished in: "
                << duration.count() << " ms" << std::endl;
      status = 1;
      break;
    }
    s *= eta;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  if (status == 0) {
    std::cout << boost::format(
                     "[EXIT] Failed to converge after %1% "
                     "iterations. (Elapsed in %2%s; Final power %3%)") %
                     max_iters % duration.count() % s
              << std::endl;
  } else if (status == 1) {
    std::cout << boost::format(
                     "[EXIT] Converge after %1% "
                     "iterations. (Elapsed in %2%s; Final power %3%)") %
                     (i + 1) % duration.count() % s
              << std::endl;
  }
  return {membership, centroids, status == 1};
}

int main() {
  matrix<double> M(4, 3);
  double data[] = {1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      M(i, j) = data[i * 3 + j];
  // s=-1 correspond to harmonic mean
  auto res = power_kmeans(M, 3, -1, 1.5, 0.01, 10, 1);
}
