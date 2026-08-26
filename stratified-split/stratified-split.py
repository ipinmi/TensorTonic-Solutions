import numpy as np

def stratified_split(X, y, test_size=0.2, seed=42):
    """
    Splits features (X) and labels (y) into training and testing sets while 
    maintaining the original class distribution (stratification).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    rng = np.random.default_rng(seed)
    
    train_indices = []
    test_indices = []
    
    # Iterate over each unique class label
    for label in np.unique(y):
        
        # np.flatnonzero(y == label) finds the indices of all samples belonging to the current class.
        # rng.permutation() randomly shuffles these indices so each split is random.
        indices = rng.permutation(np.flatnonzero(y == label))
        
        # Calculate how many samples from this specific class should go into the test set
        # Using round() to get the closest integer based on the requested test_size
        test_count = int(round(indices.size * test_size))
        
        # Safety check: If there's more than 1 sample in this class, ensure we don't 
        # accidentally put ALL of them into the test set (leaving 0 for training)
        if indices.size > 1:
            test_count = min(test_count, indices.size - 1)
            
        # Split the shuffled indices for this class based on the calculated count.
        # The first `test_count` elements go to the test set.
        test_indices.extend(indices[:test_count])
        # The remaining elements go to the training set.
        train_indices.extend(indices[test_count:])
        
    # Convert the accumulated lists back into NumPy arrays of integers.
    # sorted final indices preserve source order.
    train_indices = np.sort(np.asarray(train_indices, dtype=int))
    test_indices = np.sort(np.asarray(test_indices, dtype=int))
    
    return {
        "X_train": X[train_indices], 
        "X_test": X[test_indices],
        "y_train": y[train_indices], 
        "y_test": y[test_indices],
    }