from longitudinal.decline_detection import detect_cognitive_decline
from synthetic.scenarios import (
    stable_user,
    gradual_decline_user,
    sudden_drop_user,
    noisy_but_healthy_user
)

BASELINE = {
    "baseline_mean": 70,
    "baseline_std": 2.0
}

def test_stable_user():
    history, expected = stable_user()
    result = detect_cognitive_decline(history, BASELINE)
    assert result["status"] == expected

def test_gradual_decline_user():
    history, expected = gradual_decline_user()
    result = detect_cognitive_decline(history, BASELINE)
    assert result["status"] in ["Possible Decline", "Significant Decline"]

def test_sudden_drop_user():
    history, expected = sudden_drop_user()
    result = detect_cognitive_decline(history, BASELINE)
    assert result["status"] == expected

def test_noisy_but_healthy_user():
    history, expected = noisy_but_healthy_user()
    result = detect_cognitive_decline(history, BASELINE)
    assert result["status"] == expected

