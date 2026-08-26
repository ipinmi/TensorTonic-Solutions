import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    X_clone = np.asarray(X, dtype=float).copy()
    # 1D array
    if X_clone.ndim == 1:
        missing = np.isnan(X_clone)
        observed = X_clone[~missing]
        # If all values are missing, fill with 0.0 else with strategy
        fill = 0.0 if observed.size == 0 else float(np.mean(observed) if strategy == "mean" else np.median(observed))
        X_clone[missing] = fill
        return X_clone
    # 2D array  : moving column by column 
    for column_index in range(X_clone.shape[1]):
        column = X_clone[:, column_index]
        missing = np.isnan(column)
        observed = column[~missing]
        fill = 0.0 if observed.size == 0 else float(np.mean(observed) if strategy == "mean" else np.median(observed))
        X_clone[missing, column_index] = fill
    return X_clone