import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    # Initialize W, b
    N, D = X.shape
    w = np.zeros(D) # num of features
    b = 0.0

    for _ in range(steps):
        # matrix multiplication with @ 
        logits = X @ w + b         #Linear combination
        predictions = _sigmoid(logits)
        grad_w = X.T @ (predictions - y) / N # weighted avg across features cant use np.mean
        grad_b = np.mean(predictions - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
    