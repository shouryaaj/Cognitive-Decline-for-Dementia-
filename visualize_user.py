print("Inside visualize__user.py ...")
from visualization.chi_timeline import plot_chi_timeline
from longitudinal.decline_detection import detect_cognitive_decline
from synthetic.scenarios import gradual_decline_user

# Generate synthetic user timeline
print("Calling gradual_decline_user() ...")
user_history, _ = gradual_decline_user()

# Assume baseline already computed
print("setting up baseline...")
baseline = {
    "baseline_mean": 70,
    "baseline_std": 2.0
}

# Run decline detection
print("Calling detect_cognitive_decline() ...")
decline_result = detect_cognitive_decline(
    user_history=user_history,
    baseline=baseline
)

# Plot visualization
print("Calling plot_chi_timeline() ...")
plot_chi_timeline(
    user_history=user_history,
    baseline=baseline,
    decline_result=decline_result,
    user_id="Synthetic User"
)
