import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    #Center each feature independently 
    X_c = X - np.mean(X, axis=0)
    # Divide by N minus 1 for sample covariance
    cov = X_c.T @ X_c / (len(X) -1)
    return cov 