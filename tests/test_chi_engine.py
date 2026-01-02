from ml_core.chi_engine_v1 import compute_chi
from ml_core.risk_bands import chi_to_risk

def test_chi_engine_stability():
    scores = {
        "memory": 0.3,
        "attention": -0.4,
        "speed": 0.1
    }

    chi1 = compute_chi(scores)
    chi2 = compute_chi(scores)

    assert chi1 == chi2
    assert 0 <= chi1 <= 100
    assert chi_to_risk(chi1) in [
        "Green (Good)",
        "Yellow (Monitor)",
        "Red (High Risk)"
    ]
