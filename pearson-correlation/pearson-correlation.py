import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # numpy.corrcoef
    X = np.asarray(X, dtype=float) # shape (N,D)

    #Center each feature independently 
    X_c = X - np.mean(X, axis=0)
    # Divide by N minus 1 for sample covariance
    covariance = X_c.T @ X_c / (X.shape[0] -1)

    # Standard Deviation from diagonal covariance
    # It provides each feature variance and std can be calculated from it.
    standard_deviation = np.sqrt((np.diag(covariance)))
    # Normalize covariance by the outer product of feature standard deviations
    denominator = np.outer(standard_deviation, standard_deviation)
    
    # Correlation
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / denominator