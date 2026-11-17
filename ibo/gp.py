"""Gaussian process prediction for the IBO framework."""

import numpy as np
from scipy.linalg import solve_triangular

__all__ = ["gp_predict_fast"]


def gp_predict_fast(x_train, x_test, kernel_func, L, alpha, **kernel_args):
    """
    Calculates the mean (mu) and standard deviation (sigma) for new points.

    Variables:
    - x_train: (N, d) array of observation sites the factorisation was formed on.
    - x_test: (n, d) array of points to predict at.
    - kernel_func: the covariance function, called as
      kernel_func(x_train, x_test, **kernel_args).
    - L: Cholesky decomposition of the training covariance matrix.
    - alpha: Pre-calculated weights (L^-T * L^-1 * y).
    - v: Intermediate step for variance calculation (L^-1 * K_test_train).
    - kernel_args: the kernel parameters, typically l and nu.

    The kernel is a correlation function, so the prior variance is one and the
    returned standard deviation carries no scale: the regional estimate
    sigma_i multiplies it afterwards.
    """
    K_s = kernel_func(x_train, x_test, **kernel_args)
    mu = K_s.T @ alpha  # Predicted mean

    # Variance calculation using the Cholesky factor for speed
    v = solve_triangular(L, K_s, lower=True)
    var = 1.0 - np.sum(v ** 2, axis=0)  # Standard GP variance formula
    return mu.flatten(), np.sqrt(np.maximum(var, 0.0))
