def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    relevant_items = set(relevant)
    # Number of k recommended in relevant
    top_k_hits = sum(item in relevant_items for item in recommended[:k])
    return [float(top_k_hits / k), float(top_k_hits / len(relevant_items))]