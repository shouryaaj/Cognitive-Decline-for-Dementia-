import numpy as np
import pandas as pd
from scipy.stats import linregress

# Tunable thresholds
TREND_SLOPE_THRESHOLD = -1.0      # points per session
DROP_THRESHOLD = 10.0             # absolute CHI drop
VARIABILITY_MULTIPLIER = 2.0      # vs baseline std


def detect_trend_decline(chi_series):
    """
    Detect sustained downward trend using slope.
    """
    if len(chi_series) < 5:
        return False, None

    x = np.arange(len(chi_series))
    slope, _, _, _, _ = linregress(x, chi_series)

    if slope <= TREND_SLOPE_THRESHOLD:
        return True, f"Downward trend detected (slope={slope:.2f})"

    return False, None


def detect_drop_decline(chi_series, baseline_mean):
    """
    Detect sudden drop vs baseline.
    """
    recent = chi_series.iloc[-1]

    if baseline_mean - recent >= DROP_THRESHOLD:
        return True, f"Sudden drop detected ({baseline_mean:.1f} → {recent:.1f})"

    return False, None


def detect_variability_decline(chi_series, baseline_std):
    """
    Detect increased inconsistency.
    """
    if baseline_std == 0 or len(chi_series) < 5:
        return False, None

    recent_std = chi_series.iloc[-5:].std()

    if recent_std >= VARIABILITY_MULTIPLIER * baseline_std:
        return True, "Increased variability in recent performance"

    return False, None


def detect_cognitive_decline(user_history, baseline):
    """
    Master decline detection function.
    """
    chi_series = user_history["chi"]

    reasons = []
    triggers = 0

    trend, reason = detect_trend_decline(chi_series)
    if trend:
        triggers += 1
        reasons.append(reason)

    drop, reason = detect_drop_decline(
        chi_series, baseline["baseline_mean"]
    )
    if drop:
        triggers += 1
        reasons.append(reason)

    variability, reason = detect_variability_decline(
        chi_series, baseline["baseline_std"]
    )
    if variability:
        triggers += 1
        reasons.append(reason)

    if triggers >= 2:
        return {
            "status": "Significant Decline",
            "triggers": triggers,
            "reasons": reasons
        }

    if triggers == 1:
        return {
            "status": "Possible Decline",
            "triggers": triggers,
            "reasons": reasons
        }

    return {
        "status": "Stable",
        "triggers": 0,
        "reasons": []
    }
