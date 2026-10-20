import numpy as np
from scipy.linalg import svd

def sigmoid(x):
    """
    Sigmoid function.
    
    Parameters:
        x (ndarray): Input array.
    
    Returns:
        ndarray: Output array.
    """
    return 1 / (1 + np.exp(-x))

def fill_USVT(A, eta=0.01):
    """
    Universal Singular Value Thresholding (USVT) for missing data imputation.
    
    Parameters:
        A (ndarray): Input matrix with missing values (as NaN).
        eta (float): Thresholding parameter, should be in (0, 1).
    
    Returns:
        ndarray: Matrix with missing values imputed.
    """
    if not (0 < eta < 1):
        raise ValueError("Parameter 'eta' should be in (0, 1).")
    
    # Check if A has columns/rows full of NaN
    if np.any(np.all(np.isnan(A), axis=0)):
        print("Warning: There exists at least one column full of missing entries.")
    if np.any(np.all(np.isnan(A), axis=1)):
        print("Warning: There exists at least one row full of missing entries.")
    
    # Scaling step: Rescale to [-1, 1]
    valid_entries = A[~np.isnan(A)]
    a = np.min(valid_entries)
    b = np.max(valid_entries)
    
    if A.shape[0] > A.shape[1]:
        A = A.T
        transpose_flag = True
    else:
        transpose_flag = False
    
    m, n = A.shape
    M = (A - (a + b) / 2) / ((b - a) / 2)
    
    # Replace NaNs with 0 for proxy matrix Y
    Y = np.nan_to_num(M, nan=0.0)
    
    # Singular Value Decomposition
    U, s, Vt = svd(Y, full_matrices=False)
    
    # Proportion of observed values
    phat = np.sum(~np.isnan(A)) / A.size
    
    # Thresholding value
    thrval = (2 + eta) * np.sqrt(n * phat)
    S = s >= thrval
    
    # Define W
    if np.sum(S) == 1:
        W = (np.outer(U[:, 0], Vt[0, :]) * s[0]) / phat
    else:
        W = (U[:, S] @ np.diag(s[S]) @ Vt[S, :]) / phat
    
    # Modify Mhat and re-scale
    W = np.clip(W, -1, 1)
    Mhat = W * ((b - a) / 2) + (a + b) / 2
    
    # Handle transpose
    if transpose_flag:
        Mhat = Mhat.T
    
    return Mhat


def oppo(Z, X):
    """
    Computes the optimal value for the Orthogonal Procrustes Problem induced by matrices Z and X.

    Parameters:
        Z (numpy.ndarray): Target matrix of shape (m, n).
        X (numpy.ndarray): Source matrix of shape (m, n).

    Returns:
        error: Frobenius norm of the difference (Z - X @ R).
    """
    # Singular Value Decomposition of t(X) @ Z
    U, _, Vt = svd(X.T @ Z)
    
    # Compute R = U @ t(V)
    R = U @ Vt
    
    # Compute the error using Frobenius norm
    error = np.linalg.norm(Z - X @ R, 'fro')
    
    return error

def generate_data(n, p, d, r, rho, v_alpha, func_type,seed=2024, misspec=0.0):
    """
    Generates synthetic data.
    
    Args:
        n (int): Number of samples.
        p (int): Number of features.
        d (int): Dimensionality of the externel embeddings (E).
        r (int): Dimensionality of the latent space.
        rho (float): Sparsity parameter.
        v_alpha (float): Amount of the subject heterogeneity vector alpha.
        func_type (str): Type of the function to generate the internal embeddings (V).
        seed (int): Random seed for reproducibility.
        misspec (float): Mixing proportion a in [0, 1] for model misspecification. V is
            replaced by sqrt(1 - a) V / ||V||_F + sqrt(a) R / ||R||_F, where R is a centered
            Gaussian matrix projected onto the orthogonal complement of col(J_p E). Defaults to
            0 (no misspecification; the data are then the same as without this argument).

    Returns:
        E (numpy.ndarray): External embedding matrix of shape (p, d).
        rho_0 (float): Sparsity parameter.
        alpha (numpy.ndarray): Subject heterogeneity vector of shape (n,).
        beta (numpy.ndarray): Feature heterogeneity vector of shape (p,).
        U (numpy.ndarray): Latent subject embedding matrix of shape (n, r).
        V (numpy.ndarray): Latent feature embedding matrix of shape (p, r).
        Y (numpy.ndarray): Output binary matrix of shape (n, p).
    """
    np.random.seed(seed)
    one_p = np.ones((p, 1))
    
    # Generate Clustered E (Mimicking Embeddings)
    # Step A: Random cluster centers on the unit sphere
    k = 10      # Number of clusters (topics)
    means = np.random.randn(k, d)
    means /= np.linalg.norm(means, axis=1, keepdims=True)

    # Step B: Sample points around these centers
    # We use a mixture model approach
    mix_probs = np.ones(k) / k  # Equal probability for each cluster
    components = np.random.choice(k, size=p, p=mix_probs)

    E = []
    for idx in components:
        # Add noise to the center to create the cluster
        # Lower noise = tighter clusters (easier for RBF)
        point = means[idx] + 0.05 * np.random.randn(d) 
        # Normalize to sphere (optional, but good for embedding mimicry)
        point /= np.linalg.norm(point)
        E.append(point)
    E = np.array(E)

    # Generate the latent subject embedding matrix U
    U = np.random.randn(n, r)
    # center U column-wise
    U = U - np.mean(U, axis=0)

    # Generate latent feature embeddin matrix V
    if func_type == 'linear':
        W = np.random.randn(d, r) * 2
        V = np.dot(E, W)
    elif func_type == 'nonlinear':
        # generate two layers of nonlinear functions
        W1 = np.random.randn(d, 2*r) 
        V1 = (np.dot(E, W1))**2
        W2 = np.random.randn(2*r, r)
        V = np.tanh(np.dot(V1, W2))
    V = V - np.mean(V, axis=0)

    # Model misspecification: mix in a component orthogonal to col(J_p E)
    # (separate random generator, so that misspec = 0 leaves all other draws unchanged)
    if misspec > 0:
        rng = np.random.default_rng(seed + 1)
        Q, _ = np.linalg.qr(E - np.mean(E, axis=0))
        Z = rng.standard_normal((p, r))
        Z = Z - np.mean(Z, axis=0)
        R = Z - Q @ (Q.T @ Z)
        V = np.sqrt(1 - misspec) * V / np.linalg.norm(V) + np.sqrt(misspec) * R / np.linalg.norm(R)

    # Normalize Theta to make the squared Frobenius norm of Theta approximately in scale of n * p 
    tmp = np.sqrt( n * p / np.sum(np.dot(U, V.T) ** 2)) 
    U, S, V = np.linalg.svd(np.dot(U, V.T) * tmp, full_matrices=False)
    # Split the singular values so that U'U/n = V'V/p
    U = (n / p) ** 0.25 * U[:, :r] @ np.diag(np.sqrt(S[:r]))
    V = (p / n) ** 0.25 * V[:r, :].T @ np.diag(np.sqrt(S[:r]))

    # Generate subject heterogeneity vector alpha from uniform distribution U(1, v_alpha)
    alpha = np.random.uniform(-v_alpha, v_alpha, n).reshape(-1, 1)
    alpha = alpha  - np.mean(alpha) 

    # Compute Theta = U V^T + \rho 1_n 1_p^T + alpha 1_p^T + 1_n beta^T
    Theta = np.dot(U, V.T) + rho + alpha @ one_p.T

    # Generate binary Y based on logit Theta of shape n x p
    Y = np.random.binomial(1, sigmoid(Theta))
    
    return E, rho, alpha, U, V, Y

