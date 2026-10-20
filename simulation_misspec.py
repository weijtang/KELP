import argparse
import pickle
import os
import numpy as np
from utils import oppo, generate_data
from method import kernel_pca, usvt_initialization, PGD, PGD_soft_penalization, kernel_selection, lambda_selection

parser = argparse.ArgumentParser(description='Run the misspecification simulation (linear mapping setting).')
parser.add_argument('--n', type=int, default=200, help='Number of subjects.')
parser.add_argument('--p', type=int, default=4000, help='Number of features.')
parser.add_argument('--d', type=int, default=50, help='Dimensionality of the external embeddings.')
parser.add_argument('--r', type=int, default=8, help='Rank of the embeddings.')
parser.add_argument('--rho', type=float, default=-1.5, help='Sparsity parameter.')
parser.add_argument('--tau', type=float, default=0.5, help='Step size.')
parser.add_argument('--max_iter', type=int, default=2000, help='Maximum number of iterations.')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
parser.add_argument('-a', '--a', type=float, default=0.0, help='Misspecification proportion a in [0, 1].')
parser.add_argument('--reps', type=int, default=20, help='Number of replications.')

args = parser.parse_args()

n = args.n  # Number of samples
p = args.p  # Number of features
d = args.d   # Dimensionality of the externel embeddings (E)
r = args.r   # Dimensionality of the latent space

one_p = np.ones((p, 1))

max_iter = args.max_iter
tol = 1e-3
tau = args.tau

# Candidate kernels for data-driven selection
kernel_list = ['linear', 'rbf1', 'rbf2', 'rbf3']

# Directory to store individual replication results
output_dir = "./results_misspec_data_linear"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
setting = f'a_{args.a}'


def errors(U, V, Theta):
    return {
        'error_U': oppo(U_true, U) / np.linalg.norm(U_true, 'fro'),
        'error_V': oppo(V_true, V) / np.linalg.norm(V_true, 'fro'),
        'error_Theta': np.linalg.norm(Theta_true - Theta, 'fro') / np.linalg.norm(Theta_true, 'fro')
    }


for replication in range(args.reps):
    # Set seed
    seed = args.seed + replication
    np.random.seed(seed)
    print(f'Seed = {seed}, setting = {setting}')

    # Generate data
    E, rho_true, alpha_true, U_true, V_true, Y = generate_data(
        n, p, d, r, args.rho, v_alpha=1, func_type='linear', seed=seed,
        misspec=args.a
    )
    Theta_true = np.dot(U_true, V_true.T) + rho_true + alpha_true @ one_p.T

    # Realized fraction of V outside col(J_p E)
    Q, _ = np.linalg.qr(E - np.mean(E, axis=0))
    realized_a = np.linalg.norm(V_true - Q @ (Q.T @ V_true)) ** 2 / np.linalg.norm(V_true) ** 2

    # Baseline GD method
    rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r)
    rho_1, alpha_1, U_1, V_1, Theta_1 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init)
    results = {'baseline': errors(U_1, V_1, Theta_1)}

    # KELP with the linear kernel, warm-started from the baseline fit projected onto col(Phi)
    # (the projected USVT initializer can be too small under strong misspecification)
    Phi, q, _ = kernel_pca(E, 'linear')
    V_init = Phi @ np.linalg.solve(Phi.T @ Phi, Phi.T @ V_1)
    _, _, U_2, V_2, Theta_2 = PGD(Y, tol, max_iter, tau, rho_1, alpha_1, U_1, V_init, Phi)
    results['kelp'] = {'q': q, **errors(U_2, V_2, Theta_2)}

    # Soft penalization with the linear kernel; lambda chosen by hold-out
    best_lam, lam_losses = lambda_selection(Y, Phi, r, tol, max_iter=500, tau=tau,
                                            mask_rate=0.1, seed=args.seed)
    if best_lam == 0:
        U_3, V_3, Theta_3 = U_1, V_1, Theta_1
    elif np.isinf(best_lam):
        U_3, V_3, Theta_3 = U_2, V_2, Theta_2
    else:
        # warm-started from the (centered) baseline fit, as in lambda_selection
        _, _, U_3, V_3, Theta_3 = PGD_soft_penalization(Y, tol, max_iter, tau, rho_1, alpha_1, U_1,
                                                        V_1 - np.mean(V_1, axis=0), Phi, lam=best_lam)
    results['soft'] = {'lambda': best_lam, 'lambda_losses': lam_losses, **errors(U_3, V_3, Theta_3)}

    # KELP with data-driven kernel selection (the baseline is a candidate)
    kernel_selected = kernel_selection(Y, E, kernel_list, n, p, r, tol,
                                       max_iter=500, tau=tau,
                                       mask_rate=0.1, seed=args.seed)
    if kernel_selected == 'Baseline':
        results['selected'] = {'kernel': kernel_selected, 'q': p, **results['baseline']}
    elif kernel_selected == 'linear':
        results['selected'] = {'kernel': kernel_selected, 'q': q, **results['kelp']}
    else:
        Phi_selected, q_selected, _ = kernel_pca(E, kernel_selected)
        V_init = Phi_selected @ np.linalg.solve(Phi_selected.T @ Phi_selected, Phi_selected.T @ V_1)
        _, _, U_4, V_4, Theta_4 = PGD(Y, tol, max_iter, tau, rho_1, alpha_1, U_1, V_init, Phi_selected)
        results['selected'] = {'kernel': kernel_selected, 'q': q_selected, **errors(U_4, V_4, Theta_4)}

    # Save results for this replication
    output = {'seed': seed, 'a': args.a, 'realized_a': realized_a, **results}
    output_file = os.path.join(output_dir, f'misspec_{setting}_{replication}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    print(f"Replication {replication} results saved to {output_file}")
