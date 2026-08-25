import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    x = np.asarray(x, dtype=float)
    sample_mean = np.mean(x)
    sum = 0
    for i in x:   sum += np.square(i - sample_mean)
    var = float(sum / (len(x)- 1))
    return {"variance": var, "standard_deviation": float(np.sqrt(var))}
        