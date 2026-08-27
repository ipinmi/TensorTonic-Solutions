def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    relevant_set = set(relevant)
    # Number of k recommended in relevant
    top_k = sum(item in relevant_set for item in recommended[:k])

    return [float(top_k / k), float(top_k / len(relevant_set))]