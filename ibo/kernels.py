"""Covariance kernels for the IBO framework."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv

__all__ = ["matern_kernel"]


def matern_kernel(X1, X2, l=1.0, nu=2.5):
    """
    Computes the Matérn covariance matrix between two sets of points.

    Variables:
    - l: Length-scale (how far the influence of a point spreads).
    - nu: Smoothness parameter (0.5=rough, 1.5=medium, 2.5=smooth).
    - r: Euclidean distance between normalized points.

    The kernel is a correlation function: its diagonal is one, and the signal
    scale is supplied separately by the regional estimate sigma_i.
    """
    X1, X2 = np.atleast_2d(X1), np.atleast_2d(X2)
    r = cdist(X1 / l, X2 / l)

    if nu == 0.5:  # Absolute Exponential Kernel
        return np.exp(-r)
    elif nu == 1.5:
        sqrt3_r = np.sqrt(3) * r
        return (1 + sqrt3_r) * np.exp(-sqrt3_r)
    elif nu == 2.5:  # Standard for many BO tasks
        sqrt5_r = np.sqrt(5) * r
        return (1 + sqrt5_r + 5 / 3 * r ** 2) * np.exp(-sqrt5_r)
    elif nu == 3.5:  # Matérn 7/2 kernel
        sqrt7_r = np.sqrt(7) * r
        return (
            1
            + sqrt7_r
            + (14 / 5) * r ** 2
            + (7 * np.sqrt(7) / 15) * r ** 3
        ) * np.exp(-sqrt7_r)
    else:  # General case using Bessel functions
        sqrt_2nu_r = np.sqrt(2 * nu) * r
        K = (2 ** (1 - nu) / gamma(nu)) * (sqrt_2nu_r ** nu) * kv(nu, sqrt_2nu_r)
        K[r == 0] = 1.0  # Ensure diagonal is exactly 1.0
        return K
