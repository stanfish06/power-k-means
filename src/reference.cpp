// Reference CLI for the Python tests: runs pkm::power_kmeans on a CSV file and
// writes the results with 17 significant digits, so the tests can compare the
// Cython binding against the plain C++ build bit for bit.
//
// Usage: pkm_reference X.csv k s0 eta atol max_iters seed batch_size alpha out
// Writes out_centroids.csv, out_membership.csv and out_info.txt
// ("converged n_iter power").
#include "power_kmeans/power_kmeans.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using boost::numeric::ublas::matrix;

static matrix<double> read_csv(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    std::vector<double> row;
    std::stringstream ss(line);
    std::string cell;
    // strtod parses the shortest round-trip repr that Python writes
    while (std::getline(ss, cell, ','))
      row.push_back(std::strtod(cell.c_str(), nullptr));
    rows.push_back(row);
  }
  matrix<double> X(rows.size(), rows.empty() ? 0 : rows[0].size());
  for (std::size_t i = 0; i < X.size1(); i++) {
    if (rows[i].size() != X.size2())
      throw std::runtime_error("ragged row in " + path);
    for (std::size_t j = 0; j < X.size2(); j++)
      X(i, j) = rows[i][j];
  }
  return X;
}

static void write_csv(const std::string &path, const matrix<double> &M) {
  std::ofstream out(path);
  out << std::setprecision(17);
  for (std::size_t i = 0; i < M.size1(); i++) {
    for (std::size_t j = 0; j < M.size2(); j++)
      out << (j ? "," : "") << M(i, j);
    out << "\n";
  }
}

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "usage: " << argv[0]
              << " X.csv k s0 eta atol max_iters seed batch_size alpha out\n";
    return 2;
  }
  try {
    matrix<double> X = read_csv(argv[1]);
    pkm::Result res = pkm::power_kmeans(
        X, std::atoi(argv[2]), std::atof(argv[3]), std::atof(argv[4]),
        std::atof(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]),
        std::atoi(argv[8]), std::atof(argv[9]), 0);
    std::string out = argv[10];
    write_csv(out + "_centroids.csv", res.centroids);
    write_csv(out + "_membership.csv", res.membership);
    std::ofstream info(out + "_info.txt");
    info << std::setprecision(17) << res.converged << " " << res.n_iter << " "
         << res.power << "\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
