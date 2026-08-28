import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        shifted = x - np.max(x)
        new_values = np.exp(shifted)
        return new_values / np.sum(new_values) 

    # Keeping dimensions during row-wise reductions makes NumPy broadcast the maximum and sum across each row.
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)