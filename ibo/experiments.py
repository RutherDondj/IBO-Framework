"""Parallel workers for the IBO experiments."""

import numpy as np

from .kernels import matern_kernel
from .optimizer import bayesian_optimization_partition

__all__ = ["evaluate_single_bo"]


def evaluate_single_bo(seed, objective_function, bounds, configs, n_init, d, n_iter,
                       kernel_args, noise_var, tree_depth, sigma_estimator_func, tau=2.6,
                       n_candidates=128, min_leaf=8, return_info=False):
    """
    Generalized parallel worker to run all BO configurations for a single seed.

    Variables:
    - seed: fixes the initial design and every draw inside the loop.  All the
      configurations of one call share that design, so the comparison between
      them is paired.
    - objective_function: takes an (n, d) array and returns n values.
    - bounds: (d, 2) array; the initial design is drawn uniformly from the
      first row, so the domain is assumed the same in every coordinate.
    - configs: list of
      (label, partition_method, n_parts_fixed, sig_on, acq_name, acq_params).
    - n_init, n_iter: sizes of the initial design and of the guided budget.
    - kernel_args: the length-scale and smoothness for this objective.
    - noise_var, tree_depth, tau, min_leaf: passed to the loop.
    - sigma_estimator_func: the regional scale estimator.
    - n_candidates: per-region candidate budget, overridden by the
      `n_candidates_total` default of the loop, which equalises the inner
      optimisation effort across methods.
    - return_info: also return the partition and query history of each
      configuration.

    Returns (seed, histories) or (seed, histories, infos), where a history is
    the best value found up to each iteration, in the units of the objective.
    """
    # 1. Initialize RNG with the specific seed for a reproducible initial design
    rng = np.random.default_rng(seed)

    # Initial shared design (Assuming uniform bounds across all dimensions)
    a, b = bounds[0][0], bounds[0][1]
    x_train_init = rng.uniform(a, b, (n_init, d))
    y_train_init = objective_function(x_train_init).reshape(-1, 1)

    seed_histories, infos = {}, {}

    for label, partition_method, n_parts_fixed, sig_on, acq_name, params in configs:
        out = bayesian_optimization_partition(
            x_train_init.copy(),
            y_train_init.copy(),
            f_true=objective_function,
            bounds=bounds,
            kernel_func=matern_kernel,
            kernel_args=kernel_args,
            n_iter=n_iter,
            partition_method=partition_method,
            tree_depth=tree_depth,
            noise_var=noise_var,
            sig=sig_on,
            acq_type=acq_name,
            tau=tau,
            acq_params=params,
            n_candidates_per_region=n_candidates,
            sigma_estimator_fn=sigma_estimator_func,
            n_parts_fixed=n_parts_fixed,
            seed=seed,
            min_leaf=min_leaf,
            return_info=return_info,
        )

        if return_info:
            x_out, y_out, _, _, info = out
            infos[label] = info
        else:
            x_out, y_out, _, _ = out

        # Store "best so far"
        history = np.zeros(n_iter + 1)
        for t in range(n_iter + 1):
            history[t] = np.min(y_out[:n_init + t])

        seed_histories[label] = history

    return (seed, seed_histories, infos) if return_info else (seed, seed_histories)
