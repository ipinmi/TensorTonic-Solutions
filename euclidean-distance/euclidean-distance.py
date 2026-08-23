import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Return the Euclidean distance between x and y.
    """
    displacement = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    
    l2_norm = np.sqrt(np.sum(np.square(displacement)))

    return l2_norm