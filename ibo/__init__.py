"""IBO: Bayesian optimization for inhomogeneous functions.

The public names of the framework, so that a notebook can write

    from ibo import bayesian_optimization_partition, sigma_estimator
"""

from .kernels import matern_kernel
from .scale import sigma_estimator, get_automatic_kernel_args
from .partition import (divide_bounds, region_index, partition_tree_from_data,
                        assign_to_leaves)
from .gp import gp_predict_fast
from .optimizer import bayesian_optimization_partition
from .experiments import evaluate_single_bo
from .benchmarks import (kim1, f_egg, get_benchmarks, OFFSETS,
                         shift_benchmarks, sample_gp_function_,
                         generate_centers)
from .reporting import (plot_grouped_comparison,
                        plot_mean_ratio_across_functions,
                        plot_paired_differences, report_final_gap,
                        paired_comparison)

__all__ = [
    "matern_kernel",
    "sigma_estimator", "get_automatic_kernel_args",
    "divide_bounds", "region_index", "partition_tree_from_data",
    "assign_to_leaves",
    "gp_predict_fast",
    "bayesian_optimization_partition",
    "evaluate_single_bo",
    "kim1", "f_egg", "get_benchmarks", "OFFSETS", "shift_benchmarks",
    "sample_gp_function_", "generate_centers",
    "plot_grouped_comparison", "plot_mean_ratio_across_functions",
    "plot_paired_differences", "report_final_gap", "paired_comparison",
]
