// Demo: cluster four 3-d points into three clusters with full logging.
#include "power_kmeans/power_kmeans.hpp"

using boost::numeric::ublas::matrix;

int main() {
  matrix<double> M(4, 3);
  double data[] = {1.2, 3.4, 5.6, 7.8, 9.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      M(i, j) = data[i * 3 + j];
  // s=-1 correspond to harmonic mean
  auto res = pkm::power_kmeans(M, 3, -1, 1.5, 0.01, 10, 1, 0, 1.0, 2);
}
