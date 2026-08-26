def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    sums = {}
    counts = {}

    for cat, tgt in zip(categories, targets):
        # initialize the category sum and counts
        sums[cat] = sums.get(cat, 0.0) + tgt
        counts[cat] = counts.get(cat, 0.0) + 1
    means = {cat: sums[cat] / counts[cat] for cat in sums}
    return  [means[cat] for cat in categories]