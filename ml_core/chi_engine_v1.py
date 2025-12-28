# ml_core/chi_engine_v1.py

def compute_chi(domain_scores, weights=None):
    """
    Compute Cognitive Health Index (CHI).

    domain_scores: dict of z-scores
    returns CHI in range 0–100
    """

    if weights is None:
        weights = {k: 1.0 for k in domain_scores}

    weighted_sum = 0
    total_weight = 0

    for k, v in domain_scores.items():
        w = weights.get(k, 1.0)
        weighted_sum += w * abs(v)
        total_weight += w

    normalized = weighted_sum / total_weight
    chi = 100 - normalized * 20

    return max(0, min(100, round(chi, 2)))
