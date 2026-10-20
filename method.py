import numpy as np
from scipy.linalg import svd
from utils import fill_USVT, sigmoid
from sklearn.preprocessing import StandardScaler



# fit kernel PCA based on E: use normalized (by sample size) kernel matrix and keep the first q dim; consider three different kernels: linear, polynomial, and RBF
def compute_kernel_matrix(X, kernel_type, **params):
    """
    Computes the Gram matrix (kernel matrix) based on the kernel type and parameters.
    
    Parameters:
        X (np.ndarray): Input data matrix of shape (n, d).
        kernel_type (str): Type of kernel to use. Options: 'linear', 'polynomial', or 'rbf'.
        **params: Additional parameters for the kernel:
            - For 'polynomial': 'degree' (int, default=2), 'coef0' (float, default=0)
            - For 'rbf': 'gamma' (float, default=1.0/d where d is the number of features)
    
    Returns:
        np.ndarray: Kernel matrix (Gram matrix) of shape (n, n).
    """
    n = X.shape[0]
    
    if kernel_type == 'linear':
        return np.dot(X, X.T)
        
    elif kernel_type == 'polynomial':
        coef0 = params.get('coef0', 0)
        degree = params.get('degree', 2)
        return (np.dot(X, X.T) + coef0) ** degree
        
    elif kernel_type == 'rbf':
        gamma = params.get('gamma')
        if gamma is None:
            gamma = 1.0 / X.shape[1]
            
        # Efficient vectorization for |x-y|^2 = |x|^2 + |y|^2 - 2<x,y>
        sq_norms = np.sum(X ** 2, axis=1)
        # using [:, None] is synonymous with [:, np.newaxis] but cleaner
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * np.dot(X, X.T)
        return np.exp(-gamma * dist_sq)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def fit_kernel_pca(X, kernel_type, kernel_params={}, q=None):
    """
    Fits Kernel PCA by computing the kernel matrix, centering it, and performing eigendecomposition.
    
    The input data is standardized internally. The kernel matrix is normalized by sample size
    and centered before eigendecomposition. If q is None, it is automatically selected based
    on explained variance (eta=0.99 for linear, eta=0.95 for other kernels).
    
    Args:
        X (np.ndarray): Input data matrix of shape (n, d). Will be standardized internally.
        kernel_type (str): Type of kernel to use. Options: 'linear', 'polynomial', or 'rbf'.
        kernel_params (dict): Parameters passed to the kernel function (gamma, degree, coef0, etc).
            Defaults to empty dict.
        q (int, optional): The selected dimension (number of principal components to keep).
            If None, automatically selected based on explained variance threshold.
            Defaults to None.
        
    Returns:
        tuple: A tuple containing:
            Phi (np.ndarray): Projected data matrix of shape (n, q), containing the first q
                eigenvectors of the centered kernel matrix.
            q (int): The selected dimension (number of principal components kept).
            info (dict): Dictionary containing auxiliary information:
                - 'eigvals': All eigenvalues (sorted in descending order)
                - 'explained_variance': Cumulative explained variance ratio for the selected q components
    """
    n = X.shape[0]
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    # 1. Compute Kernel
    K = compute_kernel_matrix(X_standardized, kernel_type, **kernel_params)
    
    # 2. Normalize and Center Kernel Matrix
    K = K / n
    
    # Efficient centering: K_centered = K - 1_n K - K 1_n + 1_n K 1_n
    # Using broadcasting is faster and more memory efficient than constructing 1_n matrices
    row_means = K.mean(axis=1, keepdims=True)
    col_means = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    K_centered = K - row_means - col_means + total_mean

    # 3. Eigen Decomposition
    # eigh is faster for symmetric matrices than eig
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Numerical stability
    eigvals[eigvals < 1e-10] = 0
    
    # 4. Dimension Selection (q)
    eigvals_normalized = eigvals / np.sum(eigvals)
    
    if q is None:
        if kernel_type == 'linear':
            eta = 0.99
        else: 
            eta = 0.95
        
        cumsum = np.cumsum(eigvals_normalized)
        q = np.searchsorted(cumsum, eta) + 1

    # 5. Project data into the feature space (Phi)
    Phi = eigvecs[:, :q] 
    
    # Pack extra info if needed for analysis later
    info = {
        'eigvals': eigvals,
        'explained_variance': np.sum(eigvals_normalized[:q])
    }
    
    return Phi, q, info

def kernel_pca(E, kernel):
    """
    Fits kernel PCA using predefined kernel configurations.

    This is a convenience wrapper around fit_kernel_pca that uses predefined kernel
    configurations. Available kernels: 'linear', 'rbf1' (gamma=0.001), 'rbf2' (gamma=0.01),
    'rbf3' (gamma=0.1), 'rbf4' (gamma=1.0).

    Args:
        E (numpy.ndarray): Input data matrix of shape (n, d), where n is the number of
            samples and d is the number of features.
        kernel (str): Kernel configuration name. Options: 'linear', 'rbf1', 'rbf2', 'rbf3', 'rbf4'.

    Returns:
        tuple: A tuple containing:
            Phi (numpy.ndarray): Projected data matrix of shape (n, q), containing the first q
                eigenvectors of the centered kernel matrix.
            q (int): Number of eigenvectors (principal components) kept.
            info (dict): Dictionary containing auxiliary information:
                - 'eigvals': All eigenvalues (sorted in descending order)
                - 'explained_variance': Cumulative explained variance ratio for the selected q components
    """

    kernel_configs = {
        'linear': {'kernel_type': 'linear', 'params': {}},
        **{f'rbf{i+1}': {'kernel_type': 'rbf', 'params': {'gamma': g}} 
           for i, g in enumerate([0.001, 0.01, 0.1, 1])}
    }

    kernel_config = kernel_configs[kernel]
    Phi, q, info = fit_kernel_pca(
        E, 
        kernel_type = kernel_config['kernel_type'], 
        kernel_params = kernel_config['params']
    )

    return Phi, q, info

def usvt_initialization(Y, r, Phi=None, Mask=None, p_low=1e-5):
    """
    Initializes the latent factors using Universal Singular Value Thresholding (USVT)
    (Algorithm S1).

    Performs USVT on the observed data matrix Y to estimate the probability matrix P_hat,
    then computes initial values for rho (sparsity parameter), alpha (subject heterogeneity),
    U (subject embeddings), and V (feature embeddings). If Phi is provided, V is projected
    onto the column space of Phi (knowledge fusion). The output satisfies the balancing
    condition U0'U0/n = V0'V0/p.

    Parameters:
        Y (ndarray): Binary matrix of shape (n, p), where n is the number of subjects
            and p is the number of features.
        r (int): Rank of the embedding (number of latent dimensions).
        Phi (ndarray, optional): Kernel PCA matrix of shape (p, q) for knowledge fusion.
            If provided, V0 will be projected onto the column space of Phi. Defaults to None.
        Mask (ndarray, optional): Mask matrix of shape (n, p) indicating observed entries.
            1 indicates observed values, 0 indicates missing values. Defaults to None
            (all entries treated as observed).
        p_low (float, optional): Clipping level; P_hat is clipped to [p_low, 1 - p_low].
            Defaults to 1e-5.

    Returns:
        tuple: A tuple containing:
            rho0 (float): Initial sparsity parameter (mean of Theta_hat).
            alpha0 (ndarray): Initial subject heterogeneity vector of shape (n, 1),
                normalized to have zero mean.
            U0 (ndarray): Initial subject embedding matrix of shape (n, r).
            V0 (ndarray): Initial feature embedding matrix of shape (p, r).
                If Phi is provided, V0 is projected onto the column space of Phi.
            Theta0 (ndarray): Initial logit matrix of shape (n, p), computed as
                Theta0 = U0 @ V0.T + rho0 + alpha0 @ ones(p).T
    """
    n, p = Y.shape
    one_n = np.ones((n, 1))
    one_p = np.ones((p, 1))

    data = Y.copy()
    data = data.astype(float)

    # Step 0: For mask being 0, set those in Y to NA
    if Mask is not None:
        data[Mask == 0] = np.nan

    # Step 1: Perform USVT of Y
    P_hat = fill_USVT(data, eta=0.01)
    P_hat = np.clip(P_hat, p_low, 1 - p_low)  # Project to [p_low, 1 - p_low]
    Theta_hat = np.log(P_hat / (1 - P_hat))

    # Step 2: Compute rho0, alpha0, and beta0
    rho0 = np.mean(Theta_hat)
    alpha0 = np.mean(Theta_hat, axis=1).reshape(-1, 1) - rho0

    # Step 3: Compute the top-r component of the residual and project it to the column space of Phi
    U, S, V = svd(Theta_hat - rho0  - alpha0 @ one_p.T, full_matrices=False)
    R_r = U[:, :r] @ np.diag(S[:r]) @ V[:r, :]
    if Phi is not None:
        R_r = (R_r @ Phi) @ np.linalg.solve(Phi.T @ Phi, Phi.T)

    # Step 4: Split the singular values so that U0'U0/n = V0'V0/p
    U, S, V = svd(R_r, full_matrices=False)
    U0 = (n / p) ** 0.25 * U[:, :r] @ np.diag(np.sqrt(S[:r]))
    V0 = (p / n) ** 0.25 * V[:r, :].T @ np.diag(np.sqrt(S[:r]))

    Theta0 = np.dot(U0, V0.T) + rho0 + alpha0 @ one_p.T
    
    return rho0, alpha0, U0, V0, Theta0

def project_bounds(rho, alpha, U, V, bounds, Phi=None):
    """
    Euclidean projections onto the bounded sets C_rho, C_alpha, C_u, C_v of Algorithm 1.

    Parameters:
        rho, alpha, U, V: Current iterates; alpha must have zero mean and, if Phi is
            provided, V must lie in the column space of Phi (orthonormal columns).
        bounds (dict): Keys that are present are enforced:
            'rho': (lower, upper), i.e. (-M1, -M2).
            'alpha': M, for ||alpha||_inf <= M (keeping zero mean).
            'u': M, for max_i ||U_i||^2 <= M.
            'gamma' and 'eigvals': M and the q kernel PCA eigenvalues, for
                V = Phi diag(eigvals)^{1/2} Gamma with ||Gamma||_F^2 <= M (requires Phi).
            'v': M, for max_j ||V_j||^2 <= M (baseline without Phi).
        Phi (ndarray, optional): Kernel PCA matrix of shape (p, q).

    Returns:
        tuple: Projected (rho, alpha, U, V).
    """
    if 'rho' in bounds:
        rho = np.clip(rho, *bounds['rho'])
    if 'alpha' in bounds:
        # alpha = clip(alpha - lam, -M, M) with lam chosen by bisection so that sum(alpha) = 0
        M = bounds['alpha']
        lo, hi = alpha.min() - M, alpha.max() + M
        for _ in range(100):
            lam = (lo + hi) / 2
            if np.sum(np.clip(alpha - lam, -M, M)) > 0:
                lo = lam
            else:
                hi = lam
        alpha = np.clip(alpha - (lo + hi) / 2, -M, M)
    for key, X in [('u', U), ('v', V)]:
        if key in bounds:
            # rescale rows whose squared norm exceeds M
            norm2 = np.sum(X ** 2, axis=1, keepdims=True)
            X = X * np.minimum(1.0, np.sqrt(bounds[key] / np.maximum(norm2, 1e-300)))
            U, V = (X, V) if key == 'u' else (U, X)
    if 'gamma' in bounds:
        # ellipsoid sum_l ||B_l||^2 / mu_l <= M for B = Phi'V; project via B_l / (1 + lam / mu_l)
        M, mu = bounds['gamma'], bounds['eigvals']
        B = Phi.T @ V
        w = np.sum(B ** 2, axis=1) / mu
        if np.sum(w) > M:
            lo, hi = 0.0, 1.0
            while np.sum(w / (1 + hi / mu) ** 2) > M:
                hi *= 2
            for _ in range(100):
                lam = (lo + hi) / 2
                if np.sum(w / (1 + lam / mu) ** 2) > M:
                    lo = lam
                else:
                    hi = lam
            V = Phi @ (B / (1 + hi / mu)[:, None])
    return rho, alpha, U, V

def PGD(Y, tol, max_iter, tau, rho0, alpha0, U0, V0, Phi = None, Mask = None, bounds = None):
    """
    Performs Projected Gradient Descent (PGD) for minimizing negative log-likelihood
    (Algorithm 1).

    Optimizes the parameters (rho, alpha, U, V) of a logistic factor model by minimizing
    the negative log-likelihood plus the balancing penalty
    ||sqrt(p/n) U'U - sqrt(n/p) V'V||_F^2 / 4 using gradient descent. At each iteration, V is
    projected onto the column space of Phi (if provided) to incorporate external knowledge,
    alpha is normalized to have zero mean, and, if bounds are given, the iterates are
    projected onto the bounded sets C_rho, C_alpha, C_u, C_v. The algorithm stops when the
    normalized gradients fall below the tolerance threshold or when max_iter is reached.

    Parameters:
        Y (ndarray): Binary matrix of shape (n, p), where n is the number of subjects
            and p is the number of features.
        tol (float): Tolerance for convergence. Algorithm stops when normalized gradients
            are below this threshold.
        max_iter (int): Maximum number of iterations.
        tau (float): Base step size (eta in Algorithm 1). Actual learning rates are
            tau_rho = tau/(np), tau_alpha = tau/p, tau_u = tau/(2||V0||^2), tau_v = tau/(2||U0||^2).
        rho0 (float): Initial sparsity parameter.
        alpha0 (ndarray): Initial subject heterogeneity vector of shape (n, 1).
        U0 (ndarray): Initial subject embedding matrix of shape (n, r).
        V0 (ndarray): Initial feature embedding matrix of shape (p, r).
        Phi (ndarray, optional): Kernel PCA matrix of shape (p, q) for knowledge fusion.
            If provided, V is projected onto the column space of Phi at each iteration.
            Defaults to None.
        Mask (ndarray, optional): Mask matrix of shape (n, p) indicating observed entries.
            1 indicates observed values, 0 indicates missing values. Defaults to None
            (all entries treated as observed).
        bounds (dict, optional): Bounds of the sets C_rho, C_alpha, C_u, C_v; see
            project_bounds. Defaults to None (no bounded projections).

    Returns:
        tuple: A tuple containing:
            rho (float): Optimized sparsity parameter.
            alpha (ndarray): Optimized subject heterogeneity vector of shape (n, 1),
                normalized to have zero mean.
            U (ndarray): Optimized subject embedding matrix of shape (n, r).
            V (ndarray): Optimized feature embedding matrix of shape (p, r).
                If Phi is provided, V is in the column space of Phi.
            Theta (ndarray): Final logit matrix of shape (n, p), computed as
                Theta = U @ V.T + rho + alpha @ ones(p).T
    """
    n, p = Y.shape
    one_n = np.ones((n, 1))
    one_p = np.ones((p, 1))
    if Phi is not None:
        proj = np.linalg.solve(Phi.T @ Phi, Phi.T)

    if Mask is None:
        Mask = np.ones((n, p))

    # Learning rates
    tau_rho = tau / (n * p)
    tau_alpha = tau / (p)
    tau_u = tau / (2 * np.linalg.norm(V0, ord=2) ** 2)
    tau_v = tau / (2 * np.linalg.norm(U0, ord=2) ** 2)
    # Rescaled factors (p/n)^{1/4} U and (n/p)^{1/4} V are used in the stopping rule
    a = (p / n) ** 0.25

    # Initialize U, V, and Gamma
    rho = rho0.copy()
    alpha = alpha0.copy()
    U = U0.copy()
    V = V0.copy()
    Theta = np.dot(U, V.T) + rho + alpha @ one_p.T 
    P = sigmoid(Theta)
    
    # Perform PGD
    for _ in range(max_iter):
        # Evaluate penalty term
        g = (p / n) * U.T @ U - V.T @ V

        # calculate gradients
        grad_rho = np.sum((P - Y) * Mask)
        grad_alpha = ((P - Y) * Mask) @ one_p
        grad_U = ((P - Y) * Mask) @ V + U @ g
        grad_V = ((P - Y) * Mask).T @ U - (n / p) * V @ g

        # check if the gradient for all terms is small below tolerance
        if np.max([np.linalg.norm(np.concatenate([a * tau_u * grad_U, tau_v * grad_V / a], axis=0)) / np.linalg.norm(np.concatenate([a * U, V / a], axis=0)),
                   tau_rho * np.abs(grad_rho), 
                   tau_alpha * np.linalg.norm(grad_alpha)]) < tol:
            print('Converged at iteration', _)
            break
        elif _ % 100 == 0:
            print('Iteration:', _)
            print('Loss:', eval_loss(Y, Theta, Mask))

        # Update U and V
        rho = rho - tau_rho * grad_rho
        alpha = alpha - tau_alpha * grad_alpha
        U = U - tau_u * grad_U
        V = V - tau_v * grad_V

        # Normalizd alpha by column mean and project V to the column space of Phi
        alpha = alpha - np.mean(alpha)
        if Phi is not None:
            V = Phi @ ( proj @ V)

        # Project onto the bounded sets C_rho, C_alpha, C_u, C_v
        if bounds is not None:
            rho, alpha, U, V = project_bounds(rho, alpha, U, V, bounds, Phi)

        # Update Theta
        Theta = np.dot(U, V.T) + rho + alpha @ one_p.T
        P = sigmoid(Theta)

    return rho, alpha, U, V, Theta

def PGD_soft_penalization(Y, tol, max_iter, tau, rho0, alpha0, U0, V0, Phi = None, Mask = None, bounds = None, lam = np.inf):
    """
    Performs proximal gradient descent for the soft-penalization (soft-projection) estimator.

    Same as PGD, except that the projection of V onto the column space of Phi is replaced by
    the proximal step of the penalty (lam / 2) * ||(J_p - P_Phi) V||_F^2:
        V <- P_Phi V + (J_p - P_Phi) V / (1 + tau_v * lam),
    which shrinks rather than removes the component of V outside the semantic subspace and
    keeps V centered. lam = inf recovers PGD with Phi (KELP); lam = 0 gives the unrestricted
    GLFM with centered V.

    Parameters:
        Y, tol, max_iter, tau, rho0, alpha0, U0, V0, Mask: See PGD.
        Phi (ndarray): Kernel PCA matrix of shape (p, q) with orthonormal columns orthogonal to
            the all-ones vector (as returned by kernel_pca). If None, this is PGD without Phi.
        bounds (dict, optional): See project_bounds. The 'gamma' bound assumes V in the column
            space of Phi and is not allowed here; use 'v' (row norms) instead.
        lam (float): Penalty parameter lambda >= 0. Defaults to np.inf.

    Returns:
        tuple: (rho, alpha, U, V, Theta); see PGD.
    """
    if bounds is not None and 'gamma' in bounds:
        raise ValueError("The 'gamma' bound is not compatible with soft penalization; use 'v'.")
    n, p = Y.shape
    one_n = np.ones((n, 1))
    one_p = np.ones((p, 1))
    if Phi is not None:
        proj = np.linalg.solve(Phi.T @ Phi, Phi.T)

    if Mask is None:
        Mask = np.ones((n, p))

    # Learning rates
    tau_rho = tau / (n * p)
    tau_alpha = tau / (p)
    tau_u = tau / (2 * np.linalg.norm(V0, ord=2) ** 2)
    tau_v = tau / (2 * np.linalg.norm(U0, ord=2) ** 2)
    # Rescaled factors (p/n)^{1/4} U and (n/p)^{1/4} V are used in the stopping rule
    a = (p / n) ** 0.25

    # Initialize U, V, and Gamma
    rho = rho0.copy()
    alpha = alpha0.copy()
    U = U0.copy()
    V = V0.copy()
    Theta = np.dot(U, V.T) + rho + alpha @ one_p.T
    P = sigmoid(Theta)

    # Perform proximal gradient descent
    for _ in range(max_iter):
        # Evaluate penalty term
        g = (p / n) * U.T @ U - V.T @ V

        # calculate gradients
        grad_rho = np.sum((P - Y) * Mask)
        grad_alpha = ((P - Y) * Mask) @ one_p
        grad_U = ((P - Y) * Mask) @ V + U @ g
        grad_V = ((P - Y) * Mask).T @ U - (n / p) * V @ g

        # check if the gradient for all terms is small below tolerance
        if np.max([np.linalg.norm(np.concatenate([a * tau_u * grad_U, tau_v * grad_V / a], axis=0)) / np.linalg.norm(np.concatenate([a * U, V / a], axis=0)),
                   tau_rho * np.abs(grad_rho),
                   tau_alpha * np.linalg.norm(grad_alpha)]) < tol:
            print('Converged at iteration', _)
            break
        elif _ % 100 == 0:
            print('Iteration:', _)
            print('Loss:', eval_loss(Y, Theta, Mask))

        # Update U and V
        rho = rho - tau_rho * grad_rho
        alpha = alpha - tau_alpha * grad_alpha
        U = U - tau_u * grad_U
        V = V - tau_v * grad_V

        # Normalizd alpha by column mean and apply the proximal step of the soft penalty to V
        alpha = alpha - np.mean(alpha)
        if Phi is not None:
            V_in = Phi @ (proj @ V)
            V = V_in + (V - np.mean(V, axis=0) - V_in) / (1 + tau_v * lam)

        # Project onto the bounded sets C_rho, C_alpha, C_u, C_v
        if bounds is not None:
            rho, alpha, U, V = project_bounds(rho, alpha, U, V, bounds, Phi)

        # Update Theta
        Theta = np.dot(U, V.T) + rho + alpha @ one_p.T
        P = sigmoid(Theta)

    return rho, alpha, U, V, Theta

def eval_loss(Y, Theta, Mask):
    """
    Evaluate the negative log-likelihood loss function.

    Computes the average negative log-likelihood: -mean((Y * Theta + log(1 - sigmoid(Theta))) * Mask)
    over the observed entries (where Mask == 1).

    Parameters:
        Y (ndarray): Binary matrix of shape (n, p), where n is the number of subjects
            and p is the number of features.
        Theta (ndarray): Logit matrix of shape (n, p), typically computed as
            Theta = U @ V.T + rho + alpha @ ones(p).T
        Mask (ndarray): Mask matrix of shape (n, p) indicating observed entries.
            1 indicates observed values, 0 indicates missing values.

    Returns:
        float: Average negative log-likelihood loss over observed entries.
    """
    return -np.mean((Y * Theta + np.log(1 - sigmoid(Theta))) * Mask)

def kernel_selection(Y, E, kernel_list, n, p, r, tol, max_iter, tau, mask_rate=0.2, seed=2024, M = None):
    """
    Perform kernel selection for the knowledge-fused PGD algorithm using cross-validation.
    
    This function performs kernel selection by:
    1. Splitting the observed data into training and validation sets
    2. Running baseline PGD (without knowledge fusion) on the training set
    3. For each kernel in kernel_list, running knowledge-fused PGD starting from the
       baseline solution
    4. Evaluating each kernel on the validation set
    5. Returning the kernel with the lowest validation loss
    
    Args:
        Y (numpy.ndarray): Observed binary data matrix of shape (n, p).
        E (numpy.ndarray): External embeddings matrix of shape (n, d), where d is the
            dimension of the external feature space.
        kernel_list (list): List of kernel configuration names to evaluate.
            Options: 'linear', 'rbf1', 'rbf2', 'rbf3', 'rbf4'.
        n (int): Number of samples (subjects).
        p (int): Number of features.
        r (int): Rank of the embedding (number of latent dimensions).
        tol (float): Tolerance for convergence in PGD.
        max_iter (int): Maximum number of iterations for PGD.
        tau (float): Step size for PGD.
        mask_rate (float, optional): Proportion of observed entries to use for validation.
            The remaining (1 - mask_rate) proportion is used for training. Defaults to 0.2.
        seed (int, optional): Random seed for generating the train/validation split.
            Defaults to 2024.
        M (numpy.ndarray, optional): Binary mask matrix of shape (n, p) indicating which
            entries in Y are actually observed (1 = observed, 0 = missing). If None,
            all entries in Y are treated as observed. Defaults to None.
        
    Returns:
        str: The kernel configuration name with the lowest validation loss.
            Returns 'Baseline' if no kernel performs better than the baseline PGD.
    """

    # Store loss for all kernel list
    loss_list = {}

    # Generate random mask matrix
    np.random.seed(seed)
    Mask = np.random.binomial(1, mask_rate, size=(n, p))
    if M is not None:
        train_Mask = M * (1 - Mask)
        valid_Mask = M * Mask
    else:
        train_Mask = 1 - Mask
        valid_Mask = Mask
    print(np.mean(Mask))
      
    # Run baseline PGD
    rho_init, alpha_init, U_init, V_init, Theta_init = usvt_initialization(Y, r, Mask=train_Mask)
    print('Initial loss:', eval_loss(Y, Theta_init, valid_Mask))
    rho_1, alpha_1, U_1, V_1, Theta_1 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init, Mask=train_Mask)
    print('Loss after PGD:', eval_loss(Y, Theta_1, valid_Mask))
    # Evaluate loss
    loss_list['Baseline'] = eval_loss(Y, Theta_1, valid_Mask)

    for kernel in kernel_list:     
        # Kernel PCA
        Phi, q, info = kernel_pca(E, kernel)
        print('Kernel:', kernel, 'q:', q, 'explained_variance:', info['explained_variance'])

        if q == 0:
            continue

        # Knowledge-fused PGD
        rho_init = rho_1
        alpha_init = alpha_1
        U_init = U_1
        proj = np.linalg.solve(Phi.T @ Phi, Phi.T)
        V_init = Phi @ ( proj @ V_1 )
        _, _, _, _, Theta_2 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init, Phi, train_Mask)
        print('Loss after PGD:', eval_loss(Y, Theta_2, valid_Mask))

        # Evaluate loss
        loss_list[kernel] = eval_loss(Y, Theta_2, valid_Mask)

    # Choose one with the lowest loss
    best_kernel = min(loss_list, key=loss_list.get)

    print('Kernel selection is done! Best kernel:', best_kernel)

    return best_kernel

def lambda_selection(Y, Phi, r, tol, max_iter, tau, c_list=(1e-3, 1e-2, 1e-1, 1, 10), mask_rate=0.1, seed=2024, M = None):
    """
    Select the penalty parameter lambda of PGD_soft_penalization by hold-out validation.

    The candidates are lambda = c * ||U||_2^2 for c in c_list, where U is the baseline fit on
    the training entries (so that lambda is on the scale of the loss curvature), together with
    lambda = 0 (unrestricted GLFM) and lambda = inf (KELP). As in kernel_selection, a random
    subset of the observed entries is held out, the baseline is fit on the remaining entries,
    each candidate is fit warm-started from the baseline, and the candidate with the lowest
    hold-out loss is returned.

    Args:
        Y (numpy.ndarray): Observed binary data matrix of shape (n, p).
        Phi (numpy.ndarray): Kernel PCA matrix of shape (p, q).
        r (int): Rank of the embedding.
        tol, max_iter, tau: Passed to PGD and PGD_soft_penalization.
        c_list (sequence, optional): Multipliers of ||U||_2^2 for the finite candidates.
        mask_rate (float, optional): Proportion of observed entries held out. Defaults to 0.1.
        seed (int, optional): Random seed for the hold-out split. Defaults to 2024.
        M (numpy.ndarray, optional): Binary mask of observed entries (1 = observed).

    Returns:
        tuple: (best_lam, loss_list), where best_lam is the selected lambda (0, a finite value,
            or np.inf) and loss_list maps each candidate lambda to its hold-out loss.
    """
    n, p = Y.shape

    # Generate random mask matrix
    np.random.seed(seed)
    Mask = np.random.binomial(1, mask_rate, size=(n, p))
    if M is not None:
        train_Mask = M * (1 - Mask)
        valid_Mask = M * Mask
    else:
        train_Mask = 1 - Mask
        valid_Mask = Mask

    # Baseline fit on the training entries (lambda = 0 up to centering of V)
    rho_init, alpha_init, U_init, V_init, _ = usvt_initialization(Y, r, Mask=train_Mask)
    rho_1, alpha_1, U_1, V_1, Theta_1 = PGD(Y, tol, max_iter, tau, rho_init, alpha_init, U_init, V_init, Mask=train_Mask)
    loss_list = {0.0: eval_loss(Y, Theta_1, valid_Mask)}

    # Finite candidates and lambda = inf, warm-started from the baseline
    scale = np.linalg.norm(U_1, ord=2) ** 2
    for lam in [c * scale for c in c_list] + [np.inf]:
        _, _, _, _, Theta_2 = PGD_soft_penalization(Y, tol, max_iter, tau, rho_1, alpha_1, U_1,
                                                    V_1 - np.mean(V_1, axis=0), Phi, train_Mask, lam=lam)
        loss_list[lam] = eval_loss(Y, Theta_2, valid_Mask)
        print('lambda:', lam, 'hold-out loss:', loss_list[lam])

    best_lam = min(loss_list, key=loss_list.get)
    print('Lambda selection is done! Best lambda:', best_lam)

    return best_lam, loss_list