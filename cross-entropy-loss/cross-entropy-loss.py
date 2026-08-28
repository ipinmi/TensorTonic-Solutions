import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)  # Samples, prob distribution across all classes
    # Row pointers for every sample 
    row_indicies = np.arange(len(y_true))

    # Select probability assgined to correct class across each row 
    correct_probs = y_pred[row_indicies, y_true]
    return float(-np.mean(np.log(correct_probs)))

    # # The true class labels for your 3 samples 
    # y_true = [0, 2, 1] 

    # The row indices (0 to 2)
    #row_indices = [0, 1, 2]

    # The model's predicted probabilities for each class
    # y_pred = [
    #    [0.9, 0.05, 0.05],  # Sample 0 probabilities
    #    [0.1, 0.1,  0.8 ],  # Sample 1 probabilities
    #    [0.2, 0.7,  0.1 ]   # Sample 2 probabilities
    # ]

    # correct probs becomes selected values for [[0,0], [1,2], [2,1]]
    # [0.9, 0.8, 0.7]