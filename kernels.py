# kernels.py
import numpy as np

def rbf_kernel(x, x_prime, lengthscale=1.0, sigma_f=1.0):
    """
    RBF / Squared Exponential kernel between two points

    Args:
        x: np.ndarray, shape (d,)
        x_prime: np.ndarray, shape (d,)
        lengthscale: float
        sigma_f: float

    Returns:
        float: covariance between x and x_prime
    """
    diff = x - x_prime
    return sigma_f**2 * np.exp(-0.5 * np.dot(diff, diff) / lengthscale**2)

def matern_kernel(x, x_prime, nu=1.5, lengthscale=1.0, sigma_f=1.0):
    """
    Matern kernel (nu = 0.5, 1.5, 2.5) between two points

    Args:
        x: np.ndarray, shape (d,)
        x_prime: np.ndarray, shape (d,)
        nu: float
        lengthscale: float
        sigma_f: float

    Returns:
        float: covariance between x and x_prime
    """
    diff = x - x_prime
    r = np.sqrt(np.dot(diff, diff))

    if nu == 0.5:
        return sigma_f**2 * np.exp(-r / lengthscale)
    elif nu == 1.5:
        return sigma_f**2 * (1 + np.sqrt(3) * r / lengthscale) * np.exp(-np.sqrt(3) * r / lengthscale)
    elif nu == 2.5:
        return sigma_f**2 * (1 + np.sqrt(5) * r / lengthscale + 5*r**2/(3*lengthscale**2)) \
               * np.exp(-np.sqrt(5) * r / lengthscale)
    else:
        raise ValueError("nu must be 0.5, 1.5, or 2.5")
