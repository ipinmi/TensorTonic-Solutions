import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    # Creates an array of sequential integers from 0 up to N-1. 
    # Represents the indices of the rows. 
    indices = np.arange(N)

    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    # Divide the indices into 'k' consecutive, non-overlapping sub-arrays (folds)
    # np.array_split handles cases where N is not perfectly divisible by k 
    # (some folds will just have 1 extra element)
    folds = np.array_split(indices, k)

    # Initialize an empty list to store the train/val splits for each fold iteration.
    k_split = []

    # 'val_split' is the chunk of indices we are using for validation in this iteration.
    for idx, val_split in enumerate(folds):
        
        # Create the training set by taking all the folds EXCEPT the current one.
        # folds[:idx] gets all folds before the current one.
        # folds[idx + 1:] gets all folds after the current one.
        train = np.concatenate(folds[:idx] + folds[idx + 1:])
        
        k_split.append({
            "train_idx": train.astype(int), 
            "val_idx": val_split.astype(int)
        })
        
    return k_split