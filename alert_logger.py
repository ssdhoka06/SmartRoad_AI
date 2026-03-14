"""Alert Logger Module for SmartRoad AI.

Handles violation logging to CSV and evidence frame saving.
Part of the enforcement pipeline: when the RL agent confirms a violation,
this module captures evidence and creates a persistent log entry.

Author: Sachi (Pipeline Integration Lead)
"""

import os
import csv
import cv2
import numpy as np
from datetime import datetime


# Map action integers to human-readable labels
ACTION_LABELS = {
    0: "All Clear",
    1: "Monitor",
    2: "Violation"
}

# Default paths
DEFAULT_CSV_PATH = "violations.csv"
DEFAULT_EVIDENCE_DIR = "evidence"

# CSV column headers
CSV_HEADERS = [
    "timestamp",
    "action_label",
    "phone_detected",
    "gaze_away",
    "cigarette_detected",
    "attention_score",
    "frame_path"
]


class ViolationLogger:
    """Logs traffic violations to CSV and saves evidence frames.

    This class provides persistent violation logging with:
    - CSV-based violation records with timestamps
    - Evidence frame capture as timestamped JPEG images
    - Attention score tracking per violation event
    - Automatic directory and file creation on first use

    Attributes:
        csv_path (str): Path to the violations CSV log file.
        evidence_dir (str): Directory for saving evidence frame images.
        violation_count (int): Running count of violations logged this session.

    Usage:
        logger = ViolationLogger()
        logger.log_violation(action=2, obs_vector=obs, timestamp=ts, frame=frame)
    """

    def __init__(self, csv_path=DEFAULT_CSV_PATH, evidence_dir=DEFAULT_EVIDENCE_DIR):
        """Initialize the ViolationLogger.

        Args:
            csv_path (str): Path for the violations CSV file.
                Defaults to 'violations.csv' in the current directory.
            evidence_dir (str): Directory to save evidence frame images.
                Defaults to 'evidence/' in the current directory.
                Created automatically if it does not exist.
        """
        self.csv_path = csv_path
        self.evidence_dir = evidence_dir
        self.violation_count = 0

        # Create evidence directory if it doesn't exist
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def log_violation(self, action, obs_vector, timestamp=None, frame=None,
                      attention_score=0.0):
        """Log a violation event to the CSV file and optionally save the frame.

        Records the violation details including action taken, observation vector
        values, attention score, and path to saved evidence frame.

        Args:
            action (int): Action taken by the RL agent (0, 1, or 2).
            obs_vector (np.ndarray): 10-dimensional observation vector from obs_builder.
                Expected indices:
                    [0] phone_detected, [3] gaze_away, [4] cigarette_detected
            timestamp (str, optional): ISO-format timestamp string.
                If None, uses current datetime.
            frame (np.ndarray, optional): OpenCV BGR frame to save as evidence.
                If None, no frame is saved and frame_path is recorded as 'N/A'.
            attention_score (float): Computed attention score (0-100).
                Defaults to 0.0.

        Returns:
            str: Path to the saved evidence frame, or 'N/A' if no frame was saved.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

        action_label = ACTION_LABELS.get(action, f"Unknown({action})")

        # Extract key observations from the obs vector
        phone_detected = int(obs_vector[0]) if len(obs_vector) > 0 else 0
        gaze_away = int(obs_vector[3]) if len(obs_vector) > 3 else 0
        cigarette_detected = int(obs_vector[4]) if len(obs_vector) > 4 else 0

        # Save evidence frame if provided
        frame_path = "N/A"
        if frame is not None:
            frame_path = self.save_frame(frame, action_label, timestamp)

        # Write row to CSV
        row = [
            timestamp,
            action_label,
            phone_detected,
            gaze_away,
            cigarette_detected,
            f"{attention_score:.2f}",
            frame_path
        ]

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.violation_count += 1
        return frame_path

    def save_frame(self, frame, violation_type="Violation", timestamp=None):
        """Save an OpenCV frame as a JPEG evidence image.

        Saves the frame with a timestamped filename to the evidence directory.
        The violation type is included in the filename for easy identification.

        Args:
            frame (np.ndarray): OpenCV BGR frame (H x W x 3).
            violation_type (str): Label for the violation type, used in filename.
                Defaults to 'Violation'. Spaces are replaced with underscores.
            timestamp (str, optional): Timestamp string for the filename.
                If None, uses current datetime.

        Returns:
            str: Full path to the saved JPEG file.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

        # Sanitize violation type for filename
        safe_type = violation_type.replace(" ", "_").replace("/", "-")
        filename = f"{timestamp}_{safe_type}.jpg"
        filepath = os.path.join(self.evidence_dir, filename)

        cv2.imwrite(filepath, frame)
        return filepath

    def get_session_summary(self):
        """Get a summary of violations logged in this session.

        Returns:
            dict: Summary with keys:
                - total_violations: Number of violations logged this session
                - csv_path: Path to the CSV log file
                - evidence_dir: Path to the evidence directory
        """
        return {
            "total_violations": self.violation_count,
            "csv_path": self.csv_path,
            "evidence_dir": self.evidence_dir
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING alert_logger.py")
    print("=" * 60)

    # Create logger instance
    logger = ViolationLogger(
        csv_path="test_violations.csv",
        evidence_dir="test_evidence"
    )

    # Create a dummy observation vector (10-dim)
    dummy_obs = np.array([
        1.0,   # phone_detected
        0.91,  # phone_conf
        1.0,   # phone_near_face
        0.0,   # gaze_away
        0.0,   # cigarette_detected
        3.5,   # phone_duration
        0.0,   # gaze_duration
        0.0,   # cigarette_duration
        1.0,   # person_detected
        1.0    # driver_zone_occupied
    ], dtype=np.float32)

    # Create a dummy frame (black 480x640 image with text)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST VIOLATION FRAME", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Test 1: Log a violation with frame
    print("\nTest 1: Log violation with frame...")
    path = logger.log_violation(
        action=2,
        obs_vector=dummy_obs,
        frame=dummy_frame,
        attention_score=72.5
    )
    print(f"  Frame saved to: {path}")
    assert os.path.exists(path), "FAILED: Frame file not created"
    print("  [OK] Frame file exists")

    # Test 2: Log a monitor action without frame
    print("\nTest 2: Log monitor action without frame...")
    path = logger.log_violation(
        action=1,
        obs_vector=dummy_obs,
        attention_score=45.0
    )
    print(f"  Frame path: {path}")
    assert path == "N/A", "FAILED: Expected N/A for no frame"
    print("  [OK] No frame saved correctly")

    # Test 3: Log an all-clear action
    print("\nTest 3: Log all-clear action...")
    clear_obs = np.zeros(10, dtype=np.float32)
    clear_obs[8] = 1.0  # person detected
    clear_obs[9] = 1.0  # driver zone
    logger.log_violation(action=0, obs_vector=clear_obs, attention_score=5.0)
    print("  [OK] All clear logged")

    # Test 4: Verify CSV exists and has correct rows
    print("\nTest 4: Verify CSV file...")
    assert os.path.exists("test_violations.csv"), "FAILED: CSV not created"
    with open("test_violations.csv", "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
    print(f"  CSV has {len(rows)} rows (1 header + {len(rows)-1} data)")
    assert len(rows) == 4, f"FAILED: Expected 4 rows, got {len(rows)}"
    assert rows[0] == CSV_HEADERS, "FAILED: Headers mismatch"
    print("  [OK] CSV structure correct")

    # Test 5: Session summary
    print("\nTest 5: Session summary...")
    summary = logger.get_session_summary()
    print(f"  {summary}")
    assert summary["total_violations"] == 3
    print("  [OK] Session summary correct")

    # Cleanup test files
    os.remove("test_violations.csv")
    import shutil
    shutil.rmtree("test_evidence", ignore_errors=True)

    print("\n" + "=" * 60)
    print("ALL alert_logger.py TESTS PASSED")
    print("=" * 60)
