from synthetic.timeline_generator import generate_timeline

def stable_user():
    return generate_timeline(
        start_chi=70,
        trend=0.0,
        noise_std=1.0,
        n_sessions=10
    ), "Stable"

def gradual_decline_user():
    return generate_timeline(
        start_chi=72,
        trend=-1.5,
        noise_std=0.5,
        n_sessions=10
    ), "Possible Decline"

def sudden_drop_user():
    return generate_timeline(
        start_chi=75,
        trend=0.0,
        noise_std=0.5,
        sudden_drop=(5, 15),
        n_sessions=10
    ), "Significant Decline"

def noisy_but_healthy_user():
    return generate_timeline(
        start_chi=68,
        trend=0.0,
        noise_std=5.0,
        n_sessions=10
    ), "Stable"

