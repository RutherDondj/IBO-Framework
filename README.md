# IBO-Framework
Bayesian Optimization with Inhomogeneous Smoothness via Locally Adaptive Kernels

## Inhomogeneous Bayesian Optimization (IBO) FrameworkThis
repository contains the implementation and experimental results for the Inhomogeneous Bayesian Optimization (IBO) framework. IBO is designed to optimize black-box functions that exhibit spatially varying smoothness by using locally adaptive kernels and a partitioned modeling approach. 

## Key ContributionsSampling Allocation Theory: 

-We provide a theoretical foundation proving that to equalize acquisition function decay rates, sampling effort must scale polynomially inverse to local smoothness.  Recursive Partitioning: An adaptive tree-based strategy that subdivides the domain based on a non-stationarity ratio $\rho$, identifying regions that require higher spatial resolution.  Locally Scaled Kernels: Implementation of kernels where local scale parameters $\sigma_i$ modulate variability across subdomains.  Repository Structuresynthetic_benchmarks_IBO_F.ipynb: Reproduces results for stationary benchmarks and synthetic inhomogeneous surfaces (Figures 4, 5, and 8 in the paper).  Lasso_Diabetes_IBO_F.ipynb: Implementation of the 8-dimensional hyperparameter tuning task for LASSO regression using the LassoBench suite (Figure 11).  README.md: Project overview and theoretical context.Numerical HighlightsThe experiments demonstrate that the adaptive partitioning method ($IBO_F$) consistently outperforms standard BO in heterogeneous landscapes.  Inhomogeneous Realizations: $IBO_F$ achieves faster improvement by assigning more model flexibility to "rough" regions while maintaining broad coverage in "smooth" areas.  High-Dimensional Robustness: On the LASSO diabetes task, $IBO_F$ demonstrates efficient global exploration, converging to optimal values alongside standard benchmarks while maintaining structural robustness. 


# Inhomogeneous Bayesian Optimization (IBO) Framework

[cite_start]This repository provides the official implementation of the **Inhomogeneous Bayesian Optimization (IBO)** framework, designed for the optimization of black-box functions with spatially varying regularity[cite: 7].

---

## ## Key Contributions

* [cite_start]**Sampling Allocation Theory**: We establish a formal theoretical foundation proving that to equalize acquisition function decay rates across heterogeneous regions, the sampling effort must scale polynomially inverse to local smoothness[cite: 11, 457, 459].
* **Recursive Partitioning**: We introduce an adaptive tree-based strategy that dynamically subdivides the input domain based on a **non-stationarity ratio** $\rho$. [cite_start]This identifies and isolates regions requiring higher spatial resolution[cite: 10, 400, 414].
* [cite_start]**Locally Scaled Kernels**: Implementation of kernels where regional scale parameters $\sigma_i$ modulate variability, allowing the surrogate model to adapt to local roughness[cite: 9, 337, 340].

---

## ## Repository Structure

| File | Description |
| :--- | :--- |
| `synthetic_benchmarks_IBO_F.ipynb` | [cite_start]Reproduces results for stationary benchmarks and synthetic inhomogeneous surfaces (corresponds to Figures 4, 5, and 8 in the manuscript)[cite: 676, 677, 729]. |
| `Lasso_Diabetes_IBO_F.ipynb` | [cite_start]Implementation of the 8-dimensional hyperparameter tuning task for LASSO regression using the **LassoBench** suite (corresponds to Figure 11)[cite: 979, 980, 1007]. |
| `README.md` | Comprehensive project overview, theoretical context, and setup instructions. |

---

## ## Numerical Highlights

[cite_start]Our experiments demonstrate that the adaptive partitioning method (**$IBO_F$**) consistently outperforms standard Bayesian Optimization in heterogeneous landscapes[cite: 12, 680, 909]:

* **Inhomogeneous Realizations**: $IBO_F$ achieves a faster rate of improvement by assigning greater model flexibility to "rough" regions while maintaining broader coverage in "smooth" areas[cite: 910, 911].
* [cite_start]**High-Dimensional Robustness**: On the LASSO diabetes task, $IBO_F$ exhibits efficient global exploration and successfully converges to optimal values, maintaining structural robustness across high-dimensional search spaces[cite: 1012, 1043, 1044].

---

## ## License

This project is licensed under the **MIT License**.
