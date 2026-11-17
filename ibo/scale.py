"""Scale and length-scale estimation for the IBO framework."""

import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

__all__ = ["sigma_estimator", "get_automatic_kernel_args"]


# =====================================================================
#  1. MLE Sigma Estimator
# =====================================================================
def sigma_estimator(X, y, kernel, **kernel_args):
    """
    Maximum likelihood estimator of the scale, with a constant mean profiled
    out:

        mu_hat    = (1' K^-1 y) / (1' K^-1 1)
        sigma^2   = (y - mu_hat 1)' K^-1 (y - mu_hat 1) / N
                  = [ y' K^-1 y - (1' K^-1 y)^2 / (1' K^-1 1) ] / N

    Profiling out the mean makes the estimate invariant under f -> f + c, so
    it measures variation rather than elevation.  Without it a flat region at
    a high level produces a large sigma, and the criterion splits on how high
    the function is rather than on how rough it is.

    Variables:
    - X: (N, d) array of observation sites, all of them retained.
    - y: (N,) array of observed values; standardise them beforehand if the
      objective spans several orders of magnitude, so that the estimate
      carries the regional contrast rather than the absolute magnitude.
    - kernel: the covariance function, called as kernel(X, X, **kernel_args).
    - kernel_args: its parameters, typically l and nu.

    A small adaptive jitter is used only for numerical stability.  It is
    raised by a factor of ten whenever the factorisation fails or the
    quadratic form comes out non-positive, and the sample standard deviation
    is returned as a last resort, which keeps the scale correct even though it
    ignores the correlation structure.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).flatten()

    N = len(X)

    if N == 0:
        raise ValueError("Cannot estimate sigma with an empty dataset.")

    if len(y) != N:
        raise ValueError("X and y must have the same number of observations.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain finite values.")

    if N == 1:
        return 0.0                      # one point carries no variation

    # 1. Compute the covariance matrix using all observations
    K = kernel(X, X, **kernel_args)
    K = 0.5 * (K + K.T)

    # 2. Adaptive jitter for numerical stability
    trace_K = np.trace(K)
    jitter = 1e-8 * max(trace_K / N, 1.0)

    ones = np.ones(N)

    # 3. MLE estimation with the mean profiled out
    for _ in range(4):
        try:
            L = cholesky(K + jitter * np.eye(N), lower=True)
            Ki_y = cho_solve((L, True), y)
            Ki_1 = cho_solve((L, True), ones)

            sigma2 = (y @ Ki_y - (ones @ Ki_y) ** 2 / (ones @ Ki_1)) / N

            if np.isfinite(sigma2) and sigma2 > 0:
                return float(np.sqrt(sigma2))

            # Rounding can make the quadratic form non-positive when K is
            # ill-conditioned; more jitter usually fixes it.
            jitter *= 10

        except np.linalg.LinAlgError:
            jitter *= 10
            if jitter > 1e-4:
                warnings.warn(f"sigma_estimator jitter raised to {jitter:.1e} "
                              f"at N = {N}")

    warnings.warn(f"sigma_estimator fell back to std at N = {N}")
    return float(np.std(y))


# =====================================================================
#  2. Length-scale by maximum marginal likelihood
# =====================================================================
def get_automatic_kernel_args(functions_bench, d=2, n_points=80,
                              bounds=(-1.0, 1.0), nu=2.5):
    """
    Pre-computes the optimal length-scale (l) for each benchmark function using
    Maximum Likelihood Estimation (MLE) on an independent quasi-random Sobol
    design.

    Variables:
    - functions_bench: dict mapping a name to an objective taking an (n, d)
      array and returning n values.
    - d: input dimension of the objectives.
    - n_points: size of the Sobol design the estimate is formed on.  The
      estimate drifts with this number when the kernel is misspecified, so the
      value is a modelling choice to be stated, not a quantity to be
      discovered; n_init + n_iter matches the size the training set reaches by
      the end of a run.
    - bounds: (lower, upper) of the search domain, assumed the same in every
      coordinate.
    - nu: smoothness of the surrogate kernel, passed through to the returned
      kernel arguments so that the estimate and the kernel it is estimated for
      always agree.

    The length-scale bounds are relative to the domain width, so the same
    bracket means the same thing on [-1,1]^2 and on [-16,16]^2.  The returned
    values are intended to be fixed once and reused, not recomputed at every
    run.
    """
    kernel_args = {}
    width = bounds[1] - bounds[0]

    for name, func in functions_bench.items():
        # 1. Generate an independent quasi-random Sobol design
        # Using a fixed seed guarantees reproducible length-scales across runs
        sampler = qmc.Sobol(d=d, scramble=True, seed=42)
        sample = sampler.random(n=n_points)

        # Scale the points to the actual search domain
        X = qmc.scale(sample, bounds[0], bounds[1])

        # 2. Evaluate the objective function on the Sobol points
        # One vectorised call: the objectives accept an (n, d) array
        y = np.asarray(func(X)).ravel()

        # 3. Normalize the targets (Z-score)
        # This step is critical to stabilize the L-BFGS-B optimizer gradients
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-9)

        # 4. Define the GP Kernel at the requested smoothness
        # The bounds, relative to the domain width, prevent the length-scale
        # from collapsing to numerical noise or expanding past the domain.
        kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=0.2 * width,
            length_scale_bounds=(1e-2 * width, 5.0 * width),
            nu=nu
        )

        # 5. Fit the GP to find the optimal length-scale via MLE
        # `random_state` fixes the starting points of the L-BFGS restarts,
        # which are otherwise drawn from NumPy's global generator: without it
        # the estimate can change from one call to the next on the same data.
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b',
            alpha=1e-8,
            normalize_y=False,
            random_state=42,
        )
        gp.fit(X, y_norm)

        # 6. Extract and store the optimal length-scale
        l_opt = gp.kernel_.k2.length_scale
        kernel_args[name] = {'l': round(float(l_opt), 2), 'nu': nu}

    return kernel_args
