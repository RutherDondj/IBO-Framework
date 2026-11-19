# IBO-Framework

Bayesian Optimization for Inhomogeneous Functions via Locally Adaptive Kernels

This repository provides the official implementation of the **Inhomogeneous
Bayesian Optimization (IBO)** framework, designed for the optimization of
black-box functions with spatially varying regularity.

---

## Key contributions

* **Sampling allocation theory.** We establish that driving the regional
  acquisition suprema below a common target level is achieved by polynomially
  denser sampling in rougher subdomains, with an exponent governed by the
  ratio of the local decay rates; beyond the saturation threshold the exponent
  reduces to one and the subdomains require the same order of effort.
* **Recursive partitioning.** An adaptive tree subdivides the input domain
  along the axis of largest **non-stationarity ratio** $\eta$, the ratio of
  the regional scale estimates of the two halves, accepting a split only when
  $\eta$ exceeds a threshold $\tau$.
* **Locally scaled kernels.** The regional scale $\widehat{\sigma}_i$ multiplies
  the posterior standard deviation on each subdomain, so the acquisition
  explores where the surrogate explains the data badly.  The posterior mean is
  global and therefore continuous across the domain.

---

## Repository structure

```
ibo/                              the framework
├── kernels.py                    matern_kernel
├── scale.py                      sigma_estimator, get_automatic_kernel_args
├── partition.py                  partition_tree_from_data, divide_bounds,
│                                 region_index, assign_to_leaves
├── gp.py                         gp_predict_fast
├── optimizer.py                  bayesian_optimization_partition
├── experiments.py                evaluate_single_bo
├── benchmarks.py                 kim1, get_benchmarks, shift_benchmarks,
│                                 sample_gp_function_, generate_centers
└── reporting.py                  figures and tables

notebooks/                        one per experiment
├── 01_kim1.ipynb                 a single stationary objective
├── 02_stationary.ipynb           the ten stationary benchmarks
├── 03_inhomogeneous_single.ipynb a single inhomogeneous surface
├── 04_inhomogeneous_suite.ipynb  the ten inhomogeneous surfaces
└── 05_lasso.ipynb                LassoBench diabetes, d = 8

results/                          the pickled histories of each run
```

Notebook 04 reads the result file of notebook 02 to draw the two suites side
by side, so run 02 first.  Create `results/` before the first run.

---

## Methods compared

| Label | Partition | Regional scale |
| :--- | :--- | :--- |
| Classic BO | none | none |
| $\mathrm{IBO}_1$ | none | one global estimate |
| $\mathrm{IBO}_4$ | fixed $2\times2$ grid | one estimate per cell |
| $\mathrm{IBO}_P$ | adaptive tree | one estimate per leaf |

All four share the same kernel, the same length-scale on a given objective,
the same initial design within a seed, and the same total candidate budget for
the inner maximisation, so a partitioned run does not evaluate the acquisition
at more points than an unpartitioned one.

---

## Experimental settings

The surrogate is a Matérn kernel of smoothness $\nu = 5/2$.  Its length-scale
is estimated for each objective by maximum marginal likelihood on an
independent Sobol design of $n_{\mathrm{init}} + n_{\mathrm{iter}}$ points and
then held fixed.  Observations are standardised before the surrogate is
fitted, so that $\widehat{\sigma}_i$ reflects the local variance of the
objective rather than its absolute magnitude.  The tree has maximum depth 3, a
threshold $\tau = 1.5$ and a minimum of 16 observations per leaf, and is
rebuilt from the data every five iterations.  Each configuration is run over
30 seeds with $n_{\mathrm{init}} = 40d$ and $n_{\mathrm{iter}} = 200$.

The minimiser of each stationary benchmark is displaced from the origin before
the length-scales are fitted: five of the ten have their optimum where the
fixed grid of $\mathrm{IBO}_4$ cuts the domain, which the refinement step
would otherwise reach by projection rather than by search.

---

## Numerical highlights

| | leaves | runs with a split | $\mathrm{IBO}_P$ better than Classic BO |
| :--- | :---: | :---: | :---: |
| Ten stationary benchmarks | 4.36 | 68% | 23/40 function–acquisition pairs |
| Ten inhomogeneous surfaces | 5.17 | 99.8% | 26/40 |

The same threshold partitions almost every run on the inhomogeneous surfaces
and two thirds of the stationary ones, the latter because stationarity of an
objective does not imply a spatially constant scale: Levy, Beale, Dixon-Price
and Goldstein-Price span several orders of magnitude and are partitioned in
every run, whereas Bohachevsky and Matyas are never partitioned.

The gain comes from the partition rather than from the scale alone.  On the
inhomogeneous surfaces the median ratio of the final gap of Classic BO to that
of the variant is $0.84$ for $\mathrm{IBO}_1$, $1.02$ for $\mathrm{IBO}_4$ and
$1.55$ for $\mathrm{IBO}_P$; a value above one favours the variant.  The
advantage is significant under the improvement-based acquisitions, EI and PI,
and the methods are indistinguishable under LCB and Thompson sampling.

---

## Requirements

```
numpy
scipy
scikit-learn
matplotlib
joblib
tqdm
LassoBench        # notebook 05 only
```

Install LassoBench with

```
pip install git+https://github.com/ksehic/LassoBench.git
```

and restart the kernel afterwards: Python caches a failed import, so a package
installed after the interpreter started is not picked up without a restart.

---

## License

This project is licensed under the **MIT License**.
