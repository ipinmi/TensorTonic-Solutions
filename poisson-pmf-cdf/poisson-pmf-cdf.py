import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # probability of zero events: first exact Poisson term P(X=0)
    term = math.exp(-lam)
    cdf = term

    # The recurrence generates each following probability without recomputing a factorial.
    for i in range(1, k + 1):
        term *= lam / i
        cdf += term
    # Last term is PMF at k 
    return {"pmf": float(term), "cdf": float(cdf)}
    