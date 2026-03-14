"""
==========================================================================
SmartRoad AI — Master Integration Module (integrate.py)
==========================================================================

INTEGRATION PLAN — Module Load Order & Data Flow
-------------------------------------------------
This file orchestrates the full SmartRoad AI inference pipeline by wiring
together every independent module into one real-time loop.

Module Load Order:
    1. pipeline components (SegFormer + YOLOv8) — Layer 1 & 2
    2. obs_builder.py                           — observation vector construction
    3. attention_score.py                       — attention score computation
    4. rl_environment.py                        — RL environment (for obs space ref)
    5. stable_baselines3.PPO                    — trained RL agent (ppo_v2.zip)
    6. alert_logger.py                          — violation logging + evidence capture

Data Flow per Frame:
    Camera/Video Frame
        │
        ├──► SegFormer (every 10 frames) ──► seg_results dict
        │       keys: driver_zone (bool), steering_visible (bool)
        │
        ├──► YOLOv8 (every frame) ──► yolo_results dict
        │       keys: object_name → {bbox: [x1,y1,x2,y2], conf: float}
        │
        └──► obs_builder.build_observation(yolo_results, seg_results)
                │
                └──► 10-dim np.float32 observation vector
                        │
                        ├──► PPO agent: model.predict(obs) → action (0/1/2)
                        │
                        ├──► attention_score.compute_attention_score(tracker)
                        │       → float 0–100
                        │
                        └──► IF action == 2 (Violation):
                                alert_logger.log_violation(action, obs, ts, frame)
                                alert_logger.save_frame(frame, violation_type)

Display Overlay:
    - Top-left:  Action label (All Clear / Monitor / Violation) with color
    - Top-right: Attention Score (0–100)
    - Bottom bar: Frame count, violation count, phone/gaze durations

Key Constants:
    - SEG_INTERVAL = 10      (run SegFormer every 10th frame for performance)
    - FPS = 30               (assumed webcam frame rate)
    - MODEL_PATH = ppo_v2.zip (trained PPO agent)

Author: Sachi (Pipeline Integration Lead)
==========================================================================
"""

import os
import sys
import cv2
import torch
import numpy as np
from datetime import datetime
from PIL import Image

# ---- Layer 1: Semantic Segmentation (SegFormer) ----
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ---- Layer 2: Object Detection (YOLOv8) ----
from ultralytics import YOLO

# ---- Layer 3: RL Agent (PPO via Stable-Baselines3) ----
from stable_baselines3 import PPO

# ---- Observation Builder ----
from obs_builder import build_observation, reset_tracker, get_tracker_state

# ---- Attention Score ----
from attention_score import compute_attention_score_from_durations

# ---- Violation Logger ----
from alert_logger import ViolationLogger


# ==========================================================================
# Constants
# ==========================================================================
SEG_INTERVAL = 10          # Run SegFormer every N frames
FPS = 30                   # Assumed camera FPS
MODEL_PATH = "ppo_v2.zip"  # Path to trained PPO model (Day 3+)
FALLBACK_MODEL = "ppo_v1.zip"  # Fallback if v2 not available

# Action labels and colors (BGR for OpenCV)
ACTION_CONFIG = {
    0: {"label": "ALL CLEAR",  "color": (0, 255, 0)},    # Green
    1: {"label": "MONITOR",    "color": (0, 200, 255)},   # Yellow/Orange
    2: {"label": "VIOLATION",  "color": (0, 0, 255)},     # Red
}


# ==========================================================================
# Pipeline Helper Functions
# ==========================================================================

def load_segformer():
    """Load the SegFormer-b0 model and processor from HuggingFace.

    Returns:
        tuple: (processor, model) both ready for inference.
    """
    print("[INIT] Loading SegFormer-b0...")
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    model.eval()
    print("[INIT] SegFormer loaded.")
    return processor, model


def load_yolo():
    """Load the YOLOv8 Nano model from Ultralytics.

    Returns:
        YOLO: YOLOv8 model ready for inference.
    """
    print("[INIT] Loading YOLOv8 Nano...")
    model = YOLO("yolov8n.pt")
    print("[INIT] YOLOv8 loaded.")
    return model


def load_ppo_agent(model_path=MODEL_PATH):
    """Load a trained PPO agent from a .zip file.

    Tries MODEL_PATH first, then FALLBACK_MODEL.
    If neither exists, returns None (system runs without RL agent).

    Args:
        model_path (str): Primary path to the PPO model file.

    Returns:
        PPO or None: Loaded PPO model, or None if unavailable.
    """
    for path in [model_path, FALLBACK_MODEL]:
        if os.path.exists(path):
            print(f"[INIT] Loading PPO agent from {path}...")
            model = PPO.load(path)
            print(f"[INIT] PPO agent loaded from {path}.")
            return model
    print("[WARN] No trained PPO model found. Running without RL agent.")
    print("[WARN] Place ppo_v1.zip or ppo_v2.zip in the project directory.")
    return None


def run_segformer(frame, processor, model):
    """Run SegFormer inference on a single frame.

    Args:
        frame (np.ndarray): BGR OpenCV frame.
        processor: HuggingFace SegformerImageProcessor.
        model: HuggingFace SegformerForSemanticSegmentation.

    Returns:
        dict: Segmentation results with keys:
            - driver_zone (bool): True if person/furniture pixels detected
            - steering_visible (bool): True if relevant vehicle interior detected
            - seg_map (np.ndarray): Raw segmentation map (class per pixel)
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].numpy()
    seg_map_resized = cv2.resize(
        seg_map.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Heuristic: check for person-related and vehicle-interior classes
    unique_classes = set(np.unique(seg_map_resized).tolist())

    # ADE20K class IDs (approximate mapping for driving context):
    # 12 = person, 15 = seat/bench, 19 = curtain, 20 = chair
    # 75 = column/pillar, 135 = dashboard-like
    driver_zone = bool(unique_classes & {12, 15, 20})
    steering_visible = bool(unique_classes & {15, 20, 135}) or driver_zone

    return {
        "driver_zone": driver_zone,
        "steering_visible": steering_visible,
        "seg_map": seg_map_resized
    }


def run_yolo(frame, yolo_model):
    """Run YOLOv8 inference on a single frame.

    Args:
        frame (np.ndarray): BGR OpenCV frame.
        yolo_model: Ultralytics YOLO model.

    Returns:
        tuple: (yolo_results, annotated_frame)
            - yolo_results: dict mapping object names to {bbox, conf}
            - annotated_frame: frame with YOLO bounding box overlays
    """
    results = yolo_model(frame, verbose=False)
    annotated = results[0].plot()

    yolo_results = {}
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # Keep the highest confidence detection per class
            if cls_name not in yolo_results or conf > yolo_results[cls_name]["conf"]:
                yolo_results[cls_name] = {
                    "bbox": [int(b) for b in bbox],
                    "conf": round(conf, 4)
                }

    return yolo_results, annotated


def draw_overlay(frame, action, attention_score, frame_count, violation_count,
                 phone_dur, gaze_dur):
    """Draw status overlay on the display frame.

    Renders:
        - Top-left: Action label with color coding
        - Top-right: Attention score
        - Bottom info bar: Frame count, violation count, durations

    Args:
        frame (np.ndarray): BGR frame to draw on (modified in-place).
        action (int): Current RL agent action (0/1/2).
        attention_score (float): Current attention score (0-100).
        frame_count (int): Current frame number.
        violation_count (int): Total violations logged.
        phone_dur (float): Phone distraction duration in seconds.
        gaze_dur (float): Gaze away duration in seconds.

    Returns:
        np.ndarray: Frame with overlays drawn.
    """
    h, w = frame.shape[:2]
    config = ACTION_CONFIG.get(action, ACTION_CONFIG[0])
    label = config["label"]
    color = config["color"]

    # --- Top-left: Action label ---
    cv2.putText(frame, label, (10, 35),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

    # --- Top-right: Attention Score ---
    score_text = f"Attention: {attention_score:.1f}"
    score_color = (0, 255, 0) if attention_score < 40 else \
                  (0, 200, 255) if attention_score < 70 else (0, 0, 255)
    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, score_text, (w - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)

    # --- Bottom info bar ---
    bar_y = h - 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info_text = (f"Frame: {frame_count}  |  Violations: {violation_count}  |  "
                 f"Phone: {phone_dur:.1f}s  |  Gaze: {gaze_dur:.1f}s")
    cv2.putText(frame, info_text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def rule_based_action(obs):
    """Fallback rule-based action when no PPO model is available.

    Uses the same thresholds as the RL reward function to determine action.

    Args:
        obs (np.ndarray): 10-dim observation vector.

    Returns:
        int: Action (0=ALL_CLEAR, 1=MONITOR, 2=VIOLATION).
    """
    phone_dur = obs[5]
    gaze_dur = obs[6]

    if phone_dur > 3.0 or gaze_dur > 4.0:
        return 2  # VIOLATION
    elif phone_dur > 1.0 or gaze_dur > 1.5:
        return 1  # MONITOR
    else:
        return 0  # ALL CLEAR


# ==========================================================================
# Main Integration Loop
# ==========================================================================

def main(video_source=0):
    """Run the full SmartRoad AI integrated pipeline.

    Loads all models, opens the video source, and runs the real-time
    inference loop with RL agent decisions, attention scoring, and
    violation logging.

    Args:
        video_source: OpenCV VideoCapture source.
            - 0 for default webcam
            - filepath string for video file (e.g., 'test_data/scenario_1.mp4')

    Keyboard Controls:
        - 'q': Quit the application
        - 's': Save current frame as screenshot
    """
    print("=" * 60)
    print("SmartRoad AI — Integrated Pipeline")
    print("=" * 60)

    # ---- Load Models ----
    seg_processor, seg_model = load_segformer()
    yolo_model = load_yolo()
    ppo_agent = load_ppo_agent()
    logger = ViolationLogger()

    # ---- Open Video Source ----
    print(f"[INIT] Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {video_source}")
        sys.exit(1)
    print("[INIT] Video source opened successfully.")

    # ---- State Variables ----
    frame_count = 0
    seg_results = {"driver_zone": False, "steering_visible": True}
    seg_overlay = None

    # Color map for segmentation overlay
    np.random.seed(42)
    seg_colors = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

    # Reset observation tracker for fresh start
    reset_tracker()

    print("\n[RUN] Starting inference loop. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If reading from video file, loop or stop
                if isinstance(video_source, str):
                    print("[INFO] Video file ended.")
                    break
                else:
                    continue

            frame_count += 1

            # ---- Layer 1: SegFormer (every SEG_INTERVAL frames) ----
            if frame_count % SEG_INTERVAL == 0 or seg_overlay is None:
                seg_results = run_segformer(frame, seg_processor, seg_model)
                seg_map = seg_results["seg_map"]
                color_seg = seg_colors[seg_map]
                seg_overlay = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)

            # ---- Layer 2: YOLOv8 (every frame) ----
            yolo_results, annotated_frame = run_yolo(frame, yolo_model)

            # ---- Build Observation Vector ----
            obs = build_observation(yolo_results, seg_results)

            # ---- Layer 3: RL Agent Decision ----
            if ppo_agent is not None:
                action, _ = ppo_agent.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = rule_based_action(obs)

            # ---- Compute Attention Score ----
            tracker_state = get_tracker_state()
            attention_score = compute_attention_score_from_durations(
                phone_duration=tracker_state["phone_duration"],
                gaze_duration=tracker_state["gaze_duration"],
                cigarette_duration=tracker_state.get("cigarette_duration", 0.0)
            )

            # ---- Log Violation if Detected ----
            if action == 2:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
                logger.log_violation(
                    action=action,
                    obs_vector=obs,
                    timestamp=timestamp,
                    frame=frame,
                    attention_score=attention_score
                )

            # ---- Compose Display Frame ----
            if seg_overlay is not None:
                display = cv2.addWeighted(annotated_frame, 0.6, seg_overlay, 0.4, 0)
            else:
                display = annotated_frame.copy()

            display = draw_overlay(
                frame=display,
                action=action,
                attention_score=attention_score,
                frame_count=frame_count,
                violation_count=logger.violation_count,
                phone_dur=tracker_state["phone_duration"],
                gaze_dur=tracker_state["gaze_duration"]
            )

            # ---- Show Frame ----
            try:
                cv2.imshow("SmartRoad AI — Live", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit signal received.")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, display)
                    print(f"[INFO] Screenshot saved: {screenshot_path}")
            except cv2.error:
                # Headless environment — skip display
                pass

            # ---- Periodic Console Output ----
            if frame_count % 30 == 0:
                config = ACTION_CONFIG.get(action, ACTION_CONFIG[0])
                print(f"  Frame {frame_count:>5d}  |  "
                      f"{config['label']:<12s}  |  "
                      f"Attention: {attention_score:>5.1f}  |  "
                      f"Phone: {tracker_state['phone_duration']:.1f}s  "
                      f"Gaze: {tracker_state['gaze_duration']:.1f}s")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ---- Session Summary ----
        summary = logger.get_session_summary()
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total frames processed : {frame_count}")
        print(f"  Total violations logged: {summary['total_violations']}")
        print(f"  Violations CSV         : {summary['csv_path']}")
        print(f"  Evidence directory      : {summary['evidence_dir']}")
        print("=" * 60)


# ==========================================================================
# Entry Point
# ==========================================================================
if __name__ == "__main__":
    # Accept optional video file path as command-line argument
    # Usage: python integrate.py                    (webcam)
    #        python integrate.py test_data/scene.mp4 (video file)
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg

    main(video_source=source)
