"""The IBO optimization loop."""

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.stats import norm, qmc

from .gp import gp_predict_fast
from .partition import (assign_to_leaves, divide_bounds,
                        partition_tree_from_data)

__all__ = ["bayesian_optimization_partition"]


# =====================================================================
#   Bayesian Optimization Loop
# =====================================================================
def bayesian_optimization_partition(
        x_train, y_train, f_true, bounds,
        kernel_func=None, kernel_args=None,
        n_iter=200, partition_method="tree", tree_depth=3,
        noise_var=1e-8, sig=True,
        acq_type="ei", acq_params=None,
        n_candidates_per_region=128,
        n_candidates_total=768,         # if set, split equally across regions
        sigma_estimator_fn=None,
        n_parts_fixed=2,
        empty_region_rule="global",      # 'global' | 'optimistic'
        incumbent="global",              # 'global' | 'regional'
        n_refine_rounds=5, n_perturb=32,
        n_perturb_total=192,            # if set, split equally across regions
        refresh_every=5, tau=1.3, min_leaf=8,
        axis_rule="ratio", tree_axes=None,
        seed=None, return_info=False):
    """
    Minimise `f_true` over `bounds`.

    Variables:
    - x_train, y_train: the initial design and its evaluations, in the units
      of the objective.
    - f_true: the objective, taking an (n, d) array and returning n values.
    - bounds: (n_dim, 2) array of the search domain.
    - kernel_func, kernel_args: the covariance function and its parameters.
    - n_iter: number of guided evaluations after the initial design.
    - partition_method: 'tree' for the adaptive partition, 'fixed' for a
      regular grid of n_parts_fixed cells per axis.
    - tree_depth, tau, min_leaf, axis_rule, tree_axes: passed to
      `partition_tree_from_data`; `refresh_every` sets how often the tree is
      rebuilt from the data.
    - noise_var: initial jitter of the Cholesky factorisation, raised by a
      factor of ten on failure and capped at 1e-1.
    - sig: when False the regional scale is held at one, which gives the
      unpartitioned baseline.
    - acq_type: 'ei', 'pi', 'lcb' or 'ts'; acq_params carries xi or beta.
    - n_refine_rounds, n_perturb: the shrinking-radius search that refines the
      best candidate of each region.
    - empty_region_rule: 'optimistic' always queries a region holding no
      observation, 'global' treats it like any other.
    - seed: fixes the initial Sobol scramble and every later draw.
    - return_info: also return the partition and the query history.

    `n_candidates_per_region` and `n_perturb` are budgets *per region*, so a
    partitioned run evaluates the acquisition at `n_regions` times as many
    points as an unpartitioned one.  Part of any advantage it shows is then a
    better inner maximisation rather than a better model.  Setting
    `n_candidates_total` and `n_perturb_total` fixes the totals instead and
    divides them among the regions, which makes the comparison between
    partitioned and unpartitioned runs an equal-effort one.

    `incumbent="global"` measures improvement against the best value observed
    anywhere, so the partition enters only through the scale sigma_i: a region
    is favoured when the surrogate fits it badly, not when it has been little
    explored.  `incumbent="regional"` uses the best value observed within each
    subdomain.  The setting affects EI and PI only; LCB and Thompson sampling
    do not reference an incumbent.

    The targets are standardised before the GP is fitted.  `sigma_estimator_fn`
    returns the scale of y, so on an objective spanning several orders of
    magnitude sigma_eff = sigma_hat * sqrt(k_N) dwarfs the mean gap
    y_best - mu, and EI, LCB and Thompson sampling degenerate into space
    filling.  On standardised targets sigma_hat stays near one, so the
    regional estimate carries the regional contrast and nothing else.  The
    evaluations themselves are kept in the original units.
    """
    kernel_args = kernel_args or {"l": 0.4, "nu": 2.5}
    acq_params = acq_params or {}
    rng = np.random.default_rng(seed)

    x_train = np.atleast_2d(np.array(x_train, float))
    y_train = np.array(y_train, float).flatten()
    n_dim = x_train.shape[1]
    bounds = np.array(bounds, float)

    if partition_method == "fixed":
        leaves_bounds, _ = divide_bounds(bounds, n_parts=n_parts_fixed, mode='grid')
    elif partition_method == "tree":
        leaves_bounds = [bounds.copy()]
    else:
        raise ValueError("partition_method must be 'tree' or 'fixed'")

    sampler = qmc.Sobol(d=n_dim, scramble=True, seed=int(rng.integers(1 << 30)))
    refine = acq_type != "ts"            # TS redraws on every call
    n_leaves_hist, query_leaf_hist = [], []

    for i in range(n_iter):

        # ---- Standardised targets, recomputed as the design grows ----
        y_mean = float(np.mean(y_train))
        y_scale = float(np.std(y_train)) or 1.0
        y_work = (y_train - y_mean) / y_scale

        if partition_method == "tree" and i % refresh_every == 0:
            leaves_bounds = partition_tree_from_data(
                x_train, y_work, bounds, kernel_func, kernel_args,
                sigma_estimator_fn, max_depth=tree_depth, tau=tau,
                min_leaf=min_leaf, axes=tree_axes, axis_rule=axis_rule)

        n_regions = len(leaves_bounds)
        n_leaves_hist.append(n_regions)

        # ---- Acquisition-optimisation budget for this iteration ----
        if n_candidates_total is not None:
            n_cand = max(int(n_candidates_total // n_regions), 8)
        else:
            n_cand = n_candidates_per_region
        if n_perturb_total is not None:
            n_pert = max(int(n_perturb_total // n_regions), 4)
        else:
            n_pert = n_perturb

        sigma_per_region = np.ones(n_regions)
        y_best_per_region = np.empty(n_regions)
        region_is_empty = np.ones(n_regions, bool)

        # ---- Inline gp_fit (Robust Cholesky) ----
        K_full = kernel_func(x_train, x_train, **kernel_args)
        jitter = noise_var
        while True:
            try:
                L = cholesky(K_full + np.eye(len(K_full)) * jitter, lower=True)
                break
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > 1e-1:
                    raise np.linalg.LinAlgError("Matrix non-invertible.")
        alpha = cho_solve((L, True), y_work)

        # ---- Regional incumbents and scales ----
        g_best = float(np.min(y_work))
        # Global scale, used for regions holding too few points for a reliable
        # regional estimate.  Computed once, outside the loop, so it exists on
        # every path.
        global_sigma = (sigma_estimator_fn(x_train, y_work, kernel_func, **kernel_args)
                        if sig else 1.0)

        for r, b in enumerate(leaves_bounds):
            mask = np.all((x_train >= b[:, 0]) & (x_train <= b[:, 1]), axis=1)
            n_r = int(mask.sum())
            region_is_empty[r] = (n_r == 0)

            if sig:
                sigma_per_region[r] = (
                    sigma_estimator_fn(x_train[mask], y_work[mask],
                                       kernel_func, **kernel_args)
                    if n_r >= min_leaf else global_sigma)

            y_best_per_region[r] = (np.min(y_work[mask])
                                    if (incumbent == "regional" and n_r)
                                    else g_best)

        # ---- Inline acquisition scorer ----
        def score(P, rid, kind):
            mu, sd = gp_predict_fast(x_train, P, kernel_func, L, alpha, **kernel_args)
            sigma_eff = np.maximum(sd * sigma_per_region[rid], 1e-15)
            y_best = y_best_per_region[rid]

            if kind == 'ei':
                xi = acq_params.get('xi', 0.0)
                imp = y_best - mu - xi
                Z = imp / sigma_eff
                with np.errstate(divide="ignore", invalid="ignore"):
                    h = norm.pdf(Z) + Z * norm.cdf(Z)
                    safe = np.log(np.maximum(sigma_eff * h, 1e-300))
                    tail = np.log(sigma_eff) + norm.logpdf(Z) - 2.0 * np.log(np.abs(Z) + 1e-300)
                return np.where(Z > -6.0, safe, tail)
            elif kind == 'pi':
                xi = acq_params.get("xi", 0.0)
                return norm.logcdf((y_best - mu - xi) / sigma_eff)
            elif kind == 'lcb':
                beta = acq_params.get('beta', 1.96)
                return -(mu - beta * sigma_eff)
            elif kind == 'ts':
                return -(mu + sigma_eff * rng.standard_normal(len(mu)))
            else:
                raise ValueError(f"unknown acquisition {kind!r}")

        # ----  maximise in every region, compare peaks ----
        Xc = np.vstack([b[:, 0] + sampler.random(n_cand)
                        * (b[:, 1] - b[:, 0]) for b in leaves_bounds])
        Rc = np.repeat(np.arange(n_regions), n_cand)
        A = score(Xc, Rc, acq_type).reshape(n_regions, n_cand)
        Xm = Xc.reshape(n_regions, n_cand, n_dim)
        rows = np.arange(n_regions)
        k = np.argmax(A, axis=1)
        peak_v, peak_x = A[rows, k].astype(float), Xm[rows, k].copy()

        if refine:                       # batched shrinking-radius search
            lo = np.array([b[:, 0] for b in leaves_bounds])
            hi = np.array([b[:, 1] for b in leaves_bounds])
            rad = 0.25 * (hi - lo)
            for _ in range(n_refine_rounds):
                prop = peak_x[:, None, :] + rad[:, None, :] * rng.standard_normal(
                    (n_regions, n_pert, n_dim))
                np.clip(prop, lo[:, None, :], hi[:, None, :], out=prop)
                v = score(prop.reshape(-1, n_dim),
                          np.repeat(rows, n_pert),
                          acq_type).reshape(n_regions, n_pert)
                jj = np.argmax(v, axis=1)
                cv, cx = v[rows, jj], prop[rows, jj]
                better = cv > peak_v
                peak_v = np.where(better, cv, peak_v)
                peak_x = np.where(better[:, None], cx, peak_x)
                rad = rad * 0.5

        if empty_region_rule == "optimistic" and region_is_empty.any():
            peak_v = np.where(region_is_empty, np.inf, peak_v)
        j = int(np.argmax(peak_v))
        x_next = peak_x[j]

        # The objective and the stored history stay in the original units
        y_next = float(np.asarray(f_true(x_next.reshape(1, -1))).flatten()[0])
        x_train = np.vstack([x_train, x_next.reshape(1, -1)])
        y_train = np.append(y_train, y_next)
        query_leaf_hist.append(j)

        if (i + 1) % 8 == 0:
            sampler = qmc.Sobol(d=n_dim, scramble=True,
                                seed=int(rng.integers(1 << 30)))

    # Counts against final partition
    final_idx = assign_to_leaves(x_train, leaves_bounds)
    region_counter = np.bincount(final_idx, minlength=len(leaves_bounds))

    if return_info:
        info = {"leaves_bounds": leaves_bounds,
                "n_leaves_hist": np.array(n_leaves_hist),
                "query_leaf_hist": np.array(query_leaf_hist),
                "sigma_per_region": sigma_per_region,
                "point_leaf_index": final_idx}
        return x_train, y_train, sigma_per_region, region_counter, info

    return x_train, y_train, sigma_per_region, region_counter
