import pandas as pd
from longitudinal.decline_detection import detect_cognitive_decline

def test_detect_decline():
    history = pd.DataFrame({
        "chi": [70, 68, 66, 64, 62, 60]
    })

    baseline = {
        "baseline_mean": 70,
        "baseline_std": 1.0
    }

    result = detect_cognitive_decline(history, baseline)

    assert result["status"] in [
        "Possible Decline",
        "Significant Decline"
    ]
    assert result["triggers"] >= 1
