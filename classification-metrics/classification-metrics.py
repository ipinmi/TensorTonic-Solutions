import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)


    # Find all unique classes in the dataset (0, 1, and 2)
    classes = np.unique(y_true)

    # Build array over all classes
    # TP: Predicted class 'c' AND True class 'c'
    TP = np.array([np.sum((y_pred == c) & (y_true == c)) for c in classes], dtype=float)
    
    # FP: Predicted class 'c' BUT True is NOT 'c'
    FP = np.array([np.sum((y_pred == c) & (y_true != c)) for c in classes], dtype=float)
    
    # FN: Predicted NOT 'c' BUT True is 'c'
    FN = np.array([np.sum((y_pred != c) & (y_true == c)) for c in classes], dtype=float)
    
    # TN: Predicted NOT 'c' AND True is NOT 'c'
    TN = np.array([np.sum((y_pred != c) & (y_true != c)) for c in classes], dtype=float)

    precision_by_class = TP / np.maximum(TP + FP, 1.0)
    recall_by_class = TP / np.maximum(TP + FN, 1.0)
    f1_by_class = (2 * precision_by_class * recall_by_class) / np.maximum(precision_by_class + recall_by_class, 1e-12)
    
    # F1 Micro, Macro, Weighted, binary
    if average == "micro":
        # pool across all classes
        total_tp = float(np.sum(TP))
        total_fp = float(np.sum(FP))
        total_fn = float(np.sum(FN))
        precision = total_tp / max(total_tp + total_fp, 1.0)
        recall = total_tp / max(total_tp + total_fn, 1.0)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    elif average == "macro":
        # calculate per class and then avaerage
        precision = float(np.mean(precision_by_class))
        recall = float(np.mean(recall_by_class))
        f1 = float(np.mean(f1_by_class))
    elif average == "weighted":
        # get the count of each class actoss the true y 
        support = np.array([np.sum(y_true == c) for c in classes], dtype=float)
        weights = support / np.sum(support)
        precision = float(np.sum(weights * precision_by_class))
        recall = float(np.sum(weights * recall_by_class))
        f1 = float(np.sum(weights * f1_by_class))
    else:
        matches = np.where(classes == pos_label)[0]
        if len(matches) == 0:
            precision = recall = f1 = 0
        else:
            # take the metric values of the selected class
            index = matches[0]
            precision = float(precision_by_class[index])
            recall = float(recall_by_class[index])
            f1 = float(f1_by_class[index])

    accuracy = float(np.mean(y_pred == y_true))
    return {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        