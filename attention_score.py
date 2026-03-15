import numpy as np


# ---- Normalization Constants ----
MAX_PHONE_DURATION = 30.0
MAX_GAZE_DURATION = 30.0
MAX_CIGARETTE_DURATION = 30.0
MAX_ACTIVITY_COUNT = 10

# Weights (must sum to 1.0)
W_PHONE = 0.4
W_GAZE = 0.3
W_CIGARETTE = 0.2
W_ACTIVITY = 0.1

# Threshold recommendations
THRESHOLD_MONITOR = 60.0
THRESHOLD_VIOLATION = 80.0


def compute_attention_score(phone_duration, gaze_duration, cigarette_duration,
                            activity_count=0):
    """Compute the Attention Score from raw distraction durations.

    Args:
        phone_duration (float): Seconds phone has been detected (0-30).
        gaze_duration (float): Seconds driver gaze has been away (0-30).
        cigarette_duration (float): Seconds cigarette detected (0-30).
        activity_count (int): Number of distinct distraction events (0-10).

    Returns:
        float: Attention score in the range [0.0, 100.0].
            0 = fully attentive, 100 = maximum distraction.
    """
    norm_phone = min(phone_duration / MAX_PHONE_DURATION, 1.0)
    norm_gaze = min(gaze_duration / MAX_GAZE_DURATION, 1.0)
    norm_cig = min(cigarette_duration / MAX_CIGARETTE_DURATION, 1.0)
    norm_activity = min(activity_count / MAX_ACTIVITY_COUNT, 1.0)

    raw_score = (W_PHONE * norm_phone +
                 W_GAZE * norm_gaze +
                 W_CIGARETTE * norm_cig +
                 W_ACTIVITY * norm_activity)

    return round(raw_score * 100.0, 2)


def compute_attention_score_from_durations(phone_duration, gaze_duration,
                                           cigarette_duration=0.0,
                                           activity_count=0):
    """Convenience wrapper used by integrate.py and final_integrate.py.

    Args:
        phone_duration (float): Phone detection duration in seconds.
        gaze_duration (float): Gaze away duration in seconds.
        cigarette_duration (float): Cigarette duration in seconds. Default 0.
        activity_count (int): Activity event count. Default 0.

    Returns:
        float: Attention score (0-100).
    """
    return compute_attention_score(
        phone_duration, gaze_duration, cigarette_duration, activity_count
    )


def get_recommended_action(score):
    """Get the recommended action based on the Attention Score.

    Threshold logic:
        - score < 60  -> 0 (ALL CLEAR)
        - 60 <= score < 80 -> 1 (MONITOR)
        - score >= 80 -> 2 (VIOLATION)

    Args:
        score (float): Attention score (0-100).

    Returns:
        int: Recommended action (0, 1, or 2).
    """
    if score >= THRESHOLD_VIOLATION:
        return 2
    elif score >= THRESHOLD_MONITOR:
        return 1
    else:
        return 0


def evaluate_model(predictions, ground_truth):
    """Evaluate RL agent predictions against ground truth labels.

    Args:
        predictions (list[int]): List of predicted actions (0, 1, or 2).
        ground_truth (list[int]): List of ground truth actions (0, 1, or 2).

    Returns:
        str: Classification report as a formatted string.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    target_names = ["All Clear", "Monitor", "Violation"]
    report = classification_report(
        ground_truth, predictions, target_names=target_names, zero_division=0
    )

    cm = confusion_matrix(ground_truth, predictions, labels=[0, 1, 2])
    cm_str = "\nConfusion Matrix:\n"
    cm_str += f"{'':>15s} {'Pred_Clear':>12s} {'Pred_Monitor':>12s} {'Pred_Violation':>14s}\n"
    for i, row_name in enumerate(target_names):
        cm_str += f"{row_name:>15s} {cm[i][0]:>12d} {cm[i][1]:>12d} {cm[i][2]:>14d}\n"

    return report + cm_str


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing attention_score.py...\n")

    # Test 1: Phone 3 seconds
    score = compute_attention_score(phone_duration=3.0, gaze_duration=0.0,
                                    cigarette_duration=0.0, activity_count=1)
    print(f"Test 1 — Phone 3s: score = {score:.2f}")
    assert score > 0, "FAILED"
    print(f"  Recommended action: {get_recommended_action(score)}")

    # Test 2: No distraction
    score_clear = compute_attention_score(0, 0, 0, 0)
    print(f"\nTest 2 — No distraction: score = {score_clear:.2f}")
    assert score_clear == 0.0

    # Test 3: Maximum distraction
    score_max = compute_attention_score(30.0, 30.0, 30.0, 10)
    print(f"\nTest 3 — Max distraction: score = {score_max:.2f}")
    assert score_max == 100.0

    # Test 4: Thresholds
    print(f"\nTest 4 — Threshold checks:")
    print(f"  Score 50 -> action {get_recommended_action(50)} (expect 0)")
    print(f"  Score 65 -> action {get_recommended_action(65)} (expect 1)")
    print(f"  Score 85 -> action {get_recommended_action(85)} (expect 2)")
    assert get_recommended_action(50) == 0
    assert get_recommended_action(65) == 1
    assert get_recommended_action(85) == 2

    print("\nAll attention_score.py tests passed!")
