from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """Return the mean, median, and smallest mode."""
    x = np.asarray(x, dtype=float)
    
    counts = Counter(x)
    highest_frequency = max(counts.values())
    smallest_mode = min(value for value, count in counts.items() if count == highest_frequency)

    return {"mean":float(np.mean(x)), "median": float(np.median(x)), "mode": float(smallest_mode)} 