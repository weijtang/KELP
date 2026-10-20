import argparse
import pickle
import os
import numpy as np
from utils import oppo, generate_data
from method import kernel_pca, usvt_initialization, PGD, kernel_selection

parser = argparse.ArgumentParser(description='Run Projected GD on Simulated Data.')
parser.add_argument('--n', type=int, default=200, help='Number of subjects.')
parser.add_argument('--p', type=int, default=4000, help='Number of features.')
parser.add_argument('--d', type=int, default=50, help='Dimensionality of the external embeddings.')
parser.add_argument('--r', type=int, default=8, help='Rank of the embeddings.')
parser.add_argument('--rho', type=float, default=-1.5, help='Sparsity parameter.')
parser.add_argument('--tau', type=float, default=0.5, help='Step size.')
parser.add_argument('--max_iter', type=int, default=2000, help='Maximum number of iterations.')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
parser.add_argument('-k', '--kernel', type=str, default = 'linear', help="Kernel choice for data generation: 'linear', 'nonlinear'")

args = parser.parse_args()

n = args.n  # Number of samples
# p = args.p  # Number of features
d = args.d   # Dimensionality of the externel embeddings (E)
r = args.r   # Dimensionality of the latent space

data_kernel = args.kernel
one_n = np.ones((n, 1))


max_iter = args.max_iter
tol = 1e-3
tau = args.tau

# Kernel PCA
kernel_list = ['rbf1', 'rbf2', 'rbf3']

# Number of replications
n_replications = 20

# Directory to store individual replication results
output_dir = f"./results_p_data_{data_kernel}"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

for replication in range(n_replications):
    error_baseline = {}
    error_selected = {}
    for p in [500, 1000, 2000, 4000]:
        print(f"Running replication {replication} with p = {p}")
        # Set seed
        seed = args.seed + replication
        np.random.seed(seed)
        print(f'Seed = {seed}')
        
        # Generate data
        one_p = np.ones((p, 1))
        E, rho_true, alpha_true, U_true, V_true, Y = generate_data(
            n, p, d, r, rho=args.rho, v_alpha=1, func_type=data_kernel, seed=seed
        )
        Theta_true = np.dot(U_true, V_true.T) + rho_true + alpha_true @ one_p.T 

        # Baseline GD method
        rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r)
        _, _, U_1, V_1, Theta_1 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init)

        error_baseline[p] = {
            'error_U': oppo(U_true, U_1) / np.linalg.norm(U_true, 'fro'),
            'error_V': oppo(V_true, V_1) / np.linalg.norm(V_true, 'fro'),
            'error_Theta': np.linalg.norm(Theta_true - Theta_1, 'fro') / np.linalg.norm(Theta_true, 'fro')
        }

        # Run data-driven kernel selection
        if data_kernel == 'linear':
            kernel_selected = 'linear'
        elif data_kernel == 'nonlinear':
            kernel_selected = kernel_selection(Y, E, kernel_list, n, p, r, tol, 
                                               max_iter=500, tau=tau, 
                                               mask_rate=0.1, seed=args.seed)
        
        if kernel_selected == "Baseline":
            error_selected[p] = {
                'kernel': kernel_selected,
                'q': p,
                'error_U_kf': oppo(U_true, U_1) / np.linalg.norm(U_true, 'fro'),
                'error_V_kf': oppo(V_true, V_1) / np.linalg.norm(V_true, 'fro'),
                'error_Theta_kf': np.linalg.norm(Theta_true - Theta_1, 'fro') / np.linalg.norm(Theta_true, 'fro')
            }
        else:
            Phi_selected, q, _ = kernel_pca(E, kernel_selected)
            
            # Knowledge-fused PGD
            rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r, Phi_selected)
            _, _, U_3, V_3, Theta_3 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init, Phi_selected)

            error_selected[p] = {
                'kernel': kernel_selected,
                'q': q,
                'error_U_kf': oppo(U_true, U_3) / np.linalg.norm(U_true, 'fro'),
                'error_V_kf': oppo(V_true, V_3) / np.linalg.norm(V_true, 'fro'),
                'error_Theta_kf': np.linalg.norm(Theta_true - Theta_3, 'fro') / np.linalg.norm(Theta_true, 'fro')
            }

    # Save results for this replication
    results = {
        'seed': seed,
        'error_baseline': error_baseline,
        'error_selected': error_selected
    }
    output_file = os.path.join(output_dir, f'data_{data_kernel}_{replication}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Replication {replication} results saved to {output_file}")

