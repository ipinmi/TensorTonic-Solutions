import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    X = np.asarray(X, dtype=float)
    mins = np.min(X, axis=axis, keepdims=True)
    maxs = np.max(X, axis=axis, keepdims=True)
    data_range = maxs - mins
    safe_range = np.where(data_range > eps, data_range, 1.0)
    return (X - mins) / safe_range