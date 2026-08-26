import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y)
    if num_classes == None:
        num_classes = np.max(y) + 1

    # one hot matrix 
    matrix = np.zeros((y.size, num_classes), dtype=float)
    matrix[np.arange(y.size), y] = 1.0

    return matrix 