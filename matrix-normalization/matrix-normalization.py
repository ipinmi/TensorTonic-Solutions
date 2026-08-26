import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.asarray(matrix, dtype=float)
    # Using keepdims=True when reducing so the norm broadcasts back over the matrix.
    if norm_type == "l1":
        norms = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == "l2":
        norms = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
    elif norm_type == "max":
        norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        raise ValueError("Invalid norm_type. Expected 'l1', 'l2', or 'max'.")
    
    # Initialize the result array with zeros
    normalized = np.zeros_like(matrix)
    
    # Divide safely: only divide where the norm is not zero. 
    # Where norm is 0, the output remains 0 (from the initialization).
    np.divide(matrix, norms, out=normalized, where=(norms != 0))
    
    return normalized
        