import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)#

    expected_val = 0
    for x_i, p_x in zip(x, p):
        expected_val += x_i * p_x

    return float(expected_val)