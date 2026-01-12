# gp.py
import numpy as np

# -----------------------------
# 1. Build covariance matrix
# -----------------------------
def build_covariance_matrix(X1, X2, kernel_func, **kernel_params):
    """
    Builds covariance matrix using user-provided kernel function.

    Args:
        X1: np.ndarray, shape (n1, d)
        X2: np.ndarray, shape (n2, d)
        kernel_func: callable, kernel function(x1, x2, **kernel_params)
        kernel_params: dict, hyperparameters for kernel

    Returns:
        K: np.ndarray, shape (n1, n2)
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j], **kernel_params)
    return K

# -----------------------------
# 2. Fit GP
# -----------------------------
def fit_gp(X_train, y_train, dy_train, mean_func, mean_params, kernel_func, kernel_params):
    """
    Fits Gaussian Process to training data.

    Args:
        X_train: np.ndarray, shape (n, d)
        y_train: np.ndarray, shape (n,)
        dy_train: np.ndarray, measurement noise, shape (n,)
        mean_func: callable, mean function
        mean_params: dict, hyperparameters for mean_func
        kernel_func: callable, kernel function
        kernel_params: dict, hyperparameters for kernel_func

    Returns:
        L: Cholesky factor of K + diag(dy^2)
        alpha: solution for posterior mean computation
        mean_train: mean vector at X_train
    """
    y_train = y_train.reshape(-1,1)
    dy_train = dy_train.flatten()
    mean_train = mean_func(X_train, **mean_params)
    y_centered = y_train - mean_train

    K = build_covariance_matrix(X_train, X_train, kernel_func, **kernel_params)
    K += np.diag(dy_train**2)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))
    return L, alpha, mean_train

# -----------------------------
# 3. Predict posterior
# -----------------------------
def predict_gp(X_train, X_test, L, alpha, mean_func, mean_params, kernel_func, kernel_params):
    """
    Computes posterior mean and covariance for test points.

    Args:
        X_train: np.ndarray, training inputs
        X_test: np.ndarray, prediction inputs
        L: Cholesky factor from fit_gp
        alpha: alpha vector from fit_gp
        mean_func: callable, mean function
        mean_params: dict, hyperparameters for mean_func
        kernel_func: callable, kernel function
        kernel_params: dict, hyperparameters for kernel_func

    Returns:
        mu: posterior mean, shape (n_test,)
        cov: posterior covariance, shape (n_test, n_test)
    """
    mean_test = mean_func(X_test, **mean_params)
    K_s = build_covariance_matrix(X_train, X_test, kernel_func, **kernel_params)
    K_ss = build_covariance_matrix(X_test, X_test, kernel_func, **kernel_params) + 1e-8*np.eye(X_test.shape[0])
    
    v = np.linalg.solve(L, K_s)
    mu = mean_test + K_s.T @ alpha
    cov = K_ss - v.T @ v
    return mu.flatten(), cov

# -----------------------------
# 4. Sample from GP
# -----------------------------
def sample_gp(mu, cov, n_samples=1):
    """
    Draw samples from multivariate Gaussian (prior or posterior).

    Args:
        mu: np.ndarray, mean vector
        cov: np.ndarray, covariance matrix
        n_samples: int, number of samples

    Returns:
        samples: np.ndarray, shape (n_samples, n_points)
    """
    return np.random.multivariate_normal(mu, cov, n_samples)

# -----------------------------
# 5. Create automatic prediction grid
# -----------------------------
def create_prediction_grid(X_train, min_border=0.0, max_border=0.0, n=100):
    """
    Creates 1D prediction grid automatically based on training data, with optional borders.

    Args:
        X_train: np.ndarray, training inputs, shape (n_train, 1)
        min_border: float, extension to the left
        max_border: float, extension to the right
        n: int, number of points in the grid

    Returns:
        t: np.ndarray, shape (n, 1)
    """
    X_train_1d = X_train.flatten()
    t_min = X_train_1d.min() - min_border
    t_max = X_train_1d.max() + max_border
    t = np.linspace(t_min, t_max, n).reshape(-1,1)
    return t
