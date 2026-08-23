import numpy as np

def dot_product(x: list, y: list) -> float:
    """Return the dot product of x and y."""
    # Write code here
    x = np.asarray(x, dtype=float)
    y= np.asarray(y, dtype=float)

    sum = np.dot(x,y)
    sum = float(sum)
    
    return sum