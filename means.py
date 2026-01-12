# means.py
# Functions for Gaussian Process mean

import numpy as np

def constant_mean(x, c=0.0):
    """
    Constant mean function: returns the same value for all points.
    
    Args:
        x: np.ndarray, shape (n, d) or (d,)
        c: float, constant value
        
    Returns:
        np.ndarray, shape (n, 1)
    """
    x = np.atleast_2d(x)
    return np.full((x.shape[0], 1), c)

def linear_mean(x, w=None, b=0.0):
    """
    Linear mean function: mean(x) = w^T x + b
    
    Args:
        x: np.ndarray, shape (n, d) or (d,)
        w: np.ndarray, shape (d,), linear weights. If None, defaults to ones.
        b: float, bias term
        
    Returns:
        np.ndarray, shape (n, 1)
    """
    x = np.atleast_2d(x)
    n, d = x.shape
    if w is None:
        w = np.ones(d)
    w = np.array(w).reshape(d, 1)
    return x @ w + b
