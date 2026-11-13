# cython: language_level=3, boundscheck=False, wraparound=False
# Cython bridge between NumPy arrays and pkm::power_kmeans.

from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.utility cimport move

import numpy as np


cdef extern from "power_kmeans/power_kmeans.hpp" nogil:
    cdef cppclass Matrix "boost::numeric::ublas::matrix<double>":
        Matrix()
        Matrix(size_t, size_t) except +
        size_t size1()
        size_t size2()
        # pointer to the row-major storage; the cname makes Cython emit
        # m.data().begin()
        double* storage "data().begin"()

    cdef cppclass Result "pkm::Result":
        Matrix membership
        Matrix centroids
        bool converged
        int n_iter
        double power

    Result c_power_kmeans "pkm::power_kmeans"(
        Matrix X, int k, double s0, double eta, double atol, int max_iters,
        int seed, int batch_size, double alpha, int verbose) except +


cdef object _to_numpy(Matrix& m):
    out = np.empty((m.size1(), m.size2()), dtype=np.float64)
    cdef double[:, ::1] view = out
    if out.size:
        memcpy(&view[0, 0], m.storage(), out.size * sizeof(double))
    return out


def power_kmeans(
    const double[:, ::1] X,
    int k,
    double s0,
    double eta,
    double atol,
    int max_iters,
    int seed,
    int batch_size,
    double alpha,
    int verbose,
):
    cdef size_t n = X.shape[0], d = X.shape[1]
    cdef Matrix Xm = Matrix(n, d)
    if n * d:
        memcpy(Xm.storage(), &X[0, 0], n * d * sizeof(double))
    cdef Result res
    with nogil:
        res = c_power_kmeans(move(Xm), k, s0, eta, atol, max_iters, seed,
                             batch_size, alpha, verbose)
    return (
        _to_numpy(res.centroids),
        _to_numpy(res.membership),
        res.converged,
        res.n_iter,
        res.power,
    )
