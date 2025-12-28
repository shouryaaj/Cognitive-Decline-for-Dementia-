import numpy as np
import pandas as pd

def generate_timeline(
    start_chi=70,
    trend=0.0,
    noise_std=0.0,
    n_sessions=10,
    sudden_drop=None,
    seed=42
):
    """
    Generate a synthetic CHI timeline.

    Parameters
    ----------
    start_chi : float
        Starting CHI value
    trend : float
        Change per session (negative = decline)
    noise_std : float
        Random noise level
    sudden_drop : tuple or None
        (session_index, drop_amount)
    n_sessions : int
        Number of sessions
    """

    rng = np.random.default_rng(seed)
    chi_values = []

    for i in range(n_sessions):
        value = start_chi + trend * i

        if sudden_drop and i >= sudden_drop[0]:
            value -= sudden_drop[1]

        value += rng.normal(0, noise_std)
        chi_values.append(round(value, 2))

    return pd.DataFrame({"chi": chi_values})
