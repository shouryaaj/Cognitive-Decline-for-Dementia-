import matplotlib.pyplot as plt


def plot_chi_timeline(
    user_history,
    baseline,
    decline_result=None,
    user_id="User",
    save_path=None
):
    """
    Plot CHI over time with baseline band and decline annotation.

    Parameters
    ----------
    user_history : pandas.DataFrame
        Must contain column 'chi'
    baseline : dict
        baseline_mean, baseline_std
    decline_result : dict or None
        Output of detect_cognitive_decline()
    user_id : str
        Identifier for title
    save_path : str or None
        If provided, saves the plot instead of showing it
    """

    chi_values = user_history["chi"].values
    sessions = list(range(1, len(chi_values) + 1))

    baseline_mean = baseline["baseline_mean"]
    baseline_std = baseline["baseline_std"]

    plt.figure(figsize=(11, 5))

    # CHI curve
    plt.plot(
        sessions,
        chi_values,
        marker="o",
        linewidth=2,
        label="CHI Score"
    )

    # Baseline mean
    plt.axhline(
        baseline_mean,
        linestyle="--",
        label="Baseline Mean"
    )

    # Baseline variability band
    plt.fill_between(
        sessions,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        alpha=0.25,
        label="Baseline Variability"
    )

    # Title logic
    if decline_result and decline_result["status"] != "Stable":
        title = f"{user_id} — {decline_result['status']}"
    else:
        title = f"{user_id} — Stable"

    plt.title(title)
    plt.xlabel("Session Number")
    plt.ylabel("Cognitive Health Index (CHI)")
    plt.ylim(0, 100)
    plt.xticks(sessions)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
