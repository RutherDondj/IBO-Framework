# IBO-Framework
Bayesian Optimization for Inhomogeneous Functions via Locally Adaptive Kernels




## Inhomogeneous Bayesian Optimization (IBO) Framework

This repository provides the official implementation of the **Inhomogeneous Bayesian Optimization (IBO)** framework, designed for the optimization of black-box functions with spatially varying regularity.

---

##  Key Contributions

* **Sampling Allocation Theory**: We establish a formal theoretical foundation proving that to equalize acquisition function decay rates across heterogeneous regions, the sampling effort must scale polynomially inverse to local smoothness.
* **Recursive Partitioning**: We introduce an adaptive tree-based strategy that dynamically subdivides the input domain based on a **non-stationarity ratio** $\eta$. This identifies and isolates regions requiring higher spatial resolution.
* **Locally Scaled Kernels**: Implementation of kernels where regional scale parameters $\sigma_i$ modulate variability, allowing the surrogate model to adapt to local roughness.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `synthetic_benchmarks_IBO_F.ipynb` | Reproduces results for stationary benchmarks and synthetic inhomogeneous surfaces (corresponds to Figures 4 to 8 in the manuscript). |
| `Lasso_Diabetes_IBO_F.ipynb` | Implementation of the 8-dimensional hyperparameter tuning task for LASSO regression using the **LassoBench** suite (corresponds to Figure 11). |
| `README.md` | Comprehensive project overview, theoretical context, and setup instructions. |

---

## Numerical Highlights

Our experiments demonstrate that the adaptive partitioning method (**$IBO_F$**) consistently outperforms standard Bayesian Optimization in heterogeneous landscapes:

* **Inhomogeneous Realizations**: $IBO_F$ achieves a faster rate of improvement by assigning greater model flexibility to "rough" regions while maintaining broader coverage in "smooth" areas.
* **High-Dimensional Robustness**: On the LASSO diabetes task, $IBO_F$ exhibits efficient global exploration and successfully converges to optimal values, maintaining structural robustness across high-dimensional search spaces.

---

## License

This project is licensed under the **MIT License**.
