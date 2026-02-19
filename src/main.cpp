#include <algorithm>
#include <boost/format.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <chrono>
#include <iostream>

using namespace boost::numeric::ublas;
struct Result {
  matrix<double> membership;
  matrix<double> centroids;
  bool converged;
};
Result power_kmeans(matrix<double> X, int k, double s0, double eta, double atol,
                    int max_iters, int seed) {
  int n_obs = X.size1();
  int n_feats = X.size2();
  matrix<double> centroids(
      k, n_feats); // need initialization matrix<double> weights(n_obs, k);
  matrix<double> weights(n_obs, k);
  matrix<double> membership(n_obs, k);
  double s = s0;
  int status = 0;

  auto start = std::chrono::high_resolution_clock::now();
  int i;
  for (i = 0; i < max_iters; i++) {
    std::cout << boost::format("[INFO] Start iteration %1% with power %2%") %
                     i % s
              << std::endl;

    std::cout << "[INFO] Computing weights" << std::endl;
    auto start_wei = std::chrono::high_resolution_clock::now();
    for (int oi = 0; oi < n_obs; oi++) {
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
      _b *= _a;
      matrix_row<matrix<double>> _row_weights(weights, oi);
      matrix_row<matrix<double>> _row_membership(membership, oi);
      _row_weights = _b;
      _row_membership = _b / sum(_b);
    }
    auto end_wei = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_wei - start_wei);
    std::cout << "[SUCCESS] Weights computation finished in: "
              << duration.count() << "ms" << std::endl;

    double centroids_tot_chg_abs = 0;
    for (int ki = 0; ki < k; ki++) {
      matrix_column<matrix<double>> weights_ki(weights, ki);
      vector<double> _new_centroid_i = prod(weights_ki, X) / sum(weights_ki);
      matrix_row<matrix<double>> _old_centroid_i(centroids, ki);
      centroids_tot_chg_abs += norm_1(_new_centroid_i - _old_centroid_i);
      _old_centroid_i = _new_centroid_i;
    }
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
  if (status) {
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

int main() { std::cout << "hello" << std::endl; }
