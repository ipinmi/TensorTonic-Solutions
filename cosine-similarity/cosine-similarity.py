import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """Return the cosine similarity of a and b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    l2_norm_a = np.linalg.norm(a)
    l2_norm_b = np.linalg.norm(b)

    if l2_norm_a == 0 or l2_norm_b == 0:
        return 0.0
        
    return float(np.dot(a,b) /  (l2_norm_a * l2_norm_b))