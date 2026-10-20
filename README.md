# Knowledge-Embedded Latent Projection (KELP)

A Python implementation of the **Knowledge-Embedded Latent Projection (KELP)** method for robust representation learning. KELP incorporates external embeddings (e.g., pre-trained feature embeddings) through kernel PCA to improve embedding learning.

## Overview

KELP is a novel approach to latent factor modeling that addresses the challenges of learning embeddings from imbalanced binary matrices. The repo includes the implementation of

- **Baseline Method**: Standard generalized latent factor model (GLFM) without external knowledge
- **Knowledge-Embedded Latent Projection (KELP)**: Incorporates external embeddings via kernel PCA to improve representation learning (projected gradient descent, Algorithm 1, with the USVT initialization of Algorithm S1)
- **Data-Driven Kernel Selection**: Automatically selects the kernel (linear, RBF variants) or the baseline by hold-out validation
- **Soft-Projection Extension**: Replaces the hard subspace constraint by the penalty (λ/2)‖(J_p − P_Φ)V‖²_F, with λ chosen by hold-out validation (λ = 0 gives the GLFM and λ = ∞ gives KELP)
- **Simulations**: Scripts for evaluating performance under various settings (sample size, feature size, sparsity level, kernel choice, and misaligned semantic information), and a notebook that reproduces the simulation figures of the paper

## Installation

### Requirements

The project requires the following Python packages:

- `numpy` 1.26.4
- `scipy` 1.14.1
- `scikit-learn` 1.5.1
- `pandas` 2.2.2 (for notebooks)
- `matplotlib` 3.9.2 (for notebooks)
- `seaborn` 0.13.2 (for `simulation_summary.ipynb`)
- `jupyter` (jupyter-core 5.7.2, ipykernel 6.29.5) (for running notebooks)

You can install these packages using: `pip install numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.1 pandas==2.2.2 matplotlib==3.9.2 seaborn==0.13.2 jupyter`.

## Repo Files

```
.
├── method.py                    # Main method implementation (kernel PCA, USVT initialization, PGD, soft projection, kernel and lambda selection)
├── utils.py                     # Utility functions (USVT, sigmoid, data generation, Procrustes error)
├── demo.ipynb                   # Interactive demo notebook
├── simulation_kernel.py         # Kernel selection simulation
├── simulation_n.py              # Varying sample size simulation
├── simulation_p.py              # Varying feature size simulation
├── simulation_rho.py            # Varying sparsity simulation
├── simulation_misspec.py        # Misaligned semantic information simulation
├── simulation_summary.ipynb     # Summary and visualization of simulation results
```

## Quick Start

### Interactive Demo

The easiest way to get started is to run the interactive demo notebook:

```bash
jupyter notebook demo.ipynb
```

This notebook demonstrates:
1. **Baseline Method**: Standard latent factor model
2. **Knowledge-Embedded Method**: Using external embeddings with a specified kernel
3. **Data-Driven Kernel Selection**: Automatic kernel selection

### Basic Usage

```python
import numpy as np
from method import kernel_pca, usvt_initialization, PGD, kernel_selection
from utils import generate_data

# Generate synthetic imbalanced binary data
n, p, d, r = 200, 4000, 50, 8
E, rho_true, alpha_true, U_true, V_true, Y = generate_data(
    n, p, d, r, rho=-1.5, v_alpha=1, func_type='linear', seed=2025
)
# Note: rho=-1.5 creates imbalanced data (low probability of 1s)
tol, max_iter, tau = 1e-3, 2000, 0.5

# Method 1: Baseline (no external knowledge)
rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r)
rho, alpha, U, V, Theta = PGD(Y, tol, max_iter, tau,
                              rho_init, alpha_init, U_init, V_init)

# Method 2: Knowledge-Embedded Latent Projection (KELP) with specified kernel
Phi, q, info = kernel_pca(E, kernel='rbf2')  # or 'linear', 'rbf1', 'rbf3', 'rbf4'
rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r, Phi)
rho, alpha, U, V, Theta = PGD(Y, tol, max_iter, tau,
                              rho_init, alpha_init, U_init, V_init, Phi)

# Method 3: Data-driven kernel selection (returns a kernel name or 'Baseline')
kernel_list = ['linear', 'rbf1', 'rbf2', 'rbf3']
best_kernel = kernel_selection(Y, E, kernel_list, n, p, r, tol,
                               max_iter=500, tau=tau, mask_rate=0.1, seed=2024)
```

The soft-projection extension is fitted with `PGD_soft_penalization`, with λ chosen by `lambda_selection`:

```python
from method import PGD_soft_penalization, lambda_selection

Phi, q, info = kernel_pca(E, kernel='linear')
best_lam, lam_losses = lambda_selection(Y, Phi, r, tol, max_iter=500, tau=tau,
                                        mask_rate=0.1, seed=2024)
rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r)
rho, alpha, U, V, Theta = PGD_soft_penalization(Y, tol, max_iter, tau,
                                                rho_init, alpha_init, U_init, V_init,
                                                Phi, lam=best_lam)
```

## Parameters

### Key Parameters

- `n`: Number of subjects/samples
- `p`: Number of features
- `d`: Dimensionality of external embeddings
- `r`: Rank of latent embeddings (number of latent dimensions)
- `rho`: Sparsity parameter (controls overall probability; negative values for imbalanced data) used in data generation
- `tau`: Step size for gradient descent (η in Algorithm 1); the step sizes are τ/(np) for ρ, τ/p for α, τ/(2‖V₀‖²) for U, and τ/(2‖U₀‖²) for V
- `tol`: Convergence tolerance
- `max_iter`: Maximum number of iterations
- `Mask` (optional, `usvt_initialization`, `PGD`, `PGD_soft_penalization`): Binary matrix with 1 for observed entries; the likelihood is restricted to the observed entries (used for hold-out validation and missing-data imputation)
- `p_low` (optional, `usvt_initialization`, default `1e-5`): The USVT probability estimate is clipped to `[p_low, 1 - p_low]` before the logit transform
- `bounds` (optional, `PGD`, default `None`): Dictionary of bounds for the projections onto the constraint sets of Algorithm 1, with keys `'rho'` (lower, upper), `'alpha'`, `'u'`, and `'gamma'` with `'eigvals'` (KELP) or `'v'` (baseline); see `project_bounds` in `method.py`. With `None`, no bounded projection is applied
- `lam` (`PGD_soft_penalization`, default `np.inf`): Penalty parameter λ of the soft-projection extension

The initializer and the fitted factors satisfy the balancing condition U'U/n = V'V/p.

### Kernel Options

- `'linear'`: Linear kernel (captures linear relationships)
- `'rbf1'`: RBF kernel with gamma=0.001 (smooth, global)
- `'rbf2'`: RBF kernel with gamma=0.01 (moderate locality)
- `'rbf3'`: RBF kernel with gamma=0.1 (local)
- `'rbf4'`: RBF kernel with gamma=1.0 (very local)

## Running Simulations

The project includes several simulation scripts to evaluate performance. Each script runs 20 replications and saves one result file per replication to `./results_<setting>_data_<linear|nonlinear>/`, overwriting earlier results in that folder. Use `-k linear` or `-k nonlinear` for the mapping from the semantic embeddings to the column embeddings.

### Kernel Selection Simulation (Supplementary Figure S1)

```bash
python simulation_kernel.py --n 200 -k linear
python simulation_kernel.py --n 200 -k nonlinear
python simulation_kernel.py --n 800 -k nonlinear
```

### Varying Sample Size (Figure 1(a))

```bash
python simulation_n.py -k linear       # n = 200, 400, 800, 1600 with p = 4000
python simulation_n.py -k nonlinear
```

### Varying Feature Size (Figure 1(b))

```bash
python simulation_p.py -k linear       # p = 500-4000 with n = 200
python simulation_p.py -k nonlinear
```

### Varying Sparsity (Figure 1(c))

```bash
python simulation_rho.py -k linear     # rho = -0.5, ..., -2.7 with n = 400 and p = 4000
```

### Misaligned Semantic Information (Supplementary Figure S2)

The column embeddings mix a component orthogonal to the span of the centered semantic embeddings with proportion `a`. The script compares the baseline, KELP with the linear kernel, KELP with data-driven kernel selection, and the soft-projection extension, for one value of `a` per run:

```bash
for a in 0 0.05 0.1 0.2 0.3 0.4 0.5 0.75; do python simulation_misspec.py -a $a; done
```

The other settings are the defaults of the scripts (`--p 4000 --d 50 --r 8 --rho -1.5 --tau 0.5 --max_iter 2000 --seed 2025`).

### Summarizing Results

`simulation_summary.ipynb` reads the result folders, displays the figures of the paper, and saves them to `./figures/`.

## Citation

If you use this code in your research, please cite the associated paper:

> Tang, W., Yuan, M., Xia, Z., and Cai, T. (2026). Knowledge-Embedded Latent Projection for Robust Representation Learning. arXiv preprint arXiv:2602.16709. https://arxiv.org/abs/2602.16709

```bibtex
@article{tang2026knowledge,
  title   = {Knowledge-Embedded Latent Projection for Robust Representation Learning},
  author  = {Tang, Weijing and Yuan, Ming and Xia, Zongqi and Cai, Tianxi},
  journal = {arXiv preprint arXiv:2602.16709},
  year    = {2026}
}
```

## Contact

For questions or issues, please open an issue or contact Weijing Tang (weijingt@andrew.cmu.edu).
