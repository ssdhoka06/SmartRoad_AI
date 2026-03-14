"""
==========================================================================
SmartRoad AI — FINAL Integration Module (FROZEN)
==========================================================================
This is the production-frozen version of the SmartRoad AI pipeline.
NO FURTHER CHANGES should be made to this file after Day 4 freeze.

Version : v2.0-final
Frozen  : Day 4 — March 15, 2026
Author  : Sachi (Pipeline Integration Lead)

Pipeline: Camera → SegFormer → YOLOv8 → obs_builder → PPO Agent → Logger
==========================================================================
"""

import os
import sys
import csv
import cv2
import torch
import numpy as np
from datetime import datetime
from PIL import Image

# ---- Layer 1: Semantic Segmentation (SegFormer-b0 HuggingFace) ----
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ---- Layer 2: Object Detection (YOLOv8 Nano Ultralytics) ----
from ultralytics import YOLO

# ---- Layer 3: RL Agent (PPO via Stable-Baselines3) ----
from stable_baselines3 import PPO

# ---- Observation Builder (Ragini) ----
from obs_builder import build_observation, reset_tracker, get_tracker_state

# ---- Attention Score (Shakti) ----
from attention_score import (
    compute_attention_score_from_durations,
    get_recommended_action
)

# ---- Violation Logger (Sachi — Day 1) ----
from alert_logger import ViolationLogger


# ==========================================================================
# FROZEN Constants — Do not modify
# ==========================================================================
VERSION = "2.0-final"
SEG_INTERVAL = 10               # SegFormer runs every N frames
FPS = 30                        # Assumed camera FPS
MODEL_PATH = "ppo_v2.zip"      # Primary trained PPO agent
FALLBACK_MODEL = "ppo_v1.zip"  # Fallback PPO agent

# ADE20K class IDs relevant to driving context
ADE20K_PERSON_CLASSES = {12, 15, 20}       # person, seat/bench, chair
ADE20K_INTERIOR_CLASSES = {15, 20, 135}    # seat, chair, dashboard-like

# Action display configuration (BGR colors for OpenCV)
ACTION_CONFIG = {
    0: {"label": "ALL CLEAR",  "color": (0, 255, 0),   "priority": 0},
    1: {"label": "MONITOR",    "color": (0, 200, 255),  "priority": 1},
    2: {"label": "VIOLATION",  "color": (0, 0, 255),    "priority": 2},
}

# Segmentation color palette (deterministic)
np.random.seed(42)
SEG_COLORS = np.random.randint(0, 255, (150, 3), dtype=np.uint8)


# ==========================================================================
# Model Loaders
# ==========================================================================

def load_all_models():
    """Load all pipeline models: SegFormer, YOLOv8, PPO.

    Returns:
        dict: Dictionary with keys 'seg_processor', 'seg_model',
              'yolo', 'ppo' (PPO model or None).
    """
    print(f"[SmartRoad AI v{VERSION}] Initializing models...")

    # SegFormer
    print("  Loading SegFormer-b0 (HuggingFace)...")
    seg_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model.eval()

    # YOLOv8
    print("  Loading YOLOv8 Nano (Ultralytics)...")
    yolo = YOLO("yolov8n.pt")

    # PPO Agent
    ppo = None
    for path in [MODEL_PATH, FALLBACK_MODEL]:
        if os.path.exists(path):
            print(f"  Loading PPO agent from {path}...")
            ppo = PPO.load(path)
            break
    if ppo is None:
        print("  [WARN] No PPO model found — rule-based fallback active.")

    print("  All models loaded.\n")

    return {
        "seg_processor": seg_processor,
        "seg_model": seg_model,
        "yolo": yolo,
        "ppo": ppo
    }


# ==========================================================================
# Inference Functions
# ==========================================================================

def run_segformer(frame, processor, model):
    """Run SegFormer-b0 semantic segmentation on a frame.

    Returns:
        tuple: (seg_results_dict, seg_overlay_bgr)
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = outputs.logits.argmax(dim=1)[0].numpy()
    seg_resized = cv2.resize(
        seg_map.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    unique = set(np.unique(seg_resized).tolist())
    driver_zone = bool(unique & ADE20K_PERSON_CLASSES)
    steering_visible = bool(unique & ADE20K_INTERIOR_CLASSES) or driver_zone

    seg_results = {
        "driver_zone": driver_zone,
        "steering_visible": steering_visible
    }

    color_seg = SEG_COLORS[seg_resized]
    seg_overlay = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)

    return seg_results, seg_overlay


def run_yolo(frame, yolo_model):
    """Run YOLOv8 Nano object detection on a frame.

    Returns:
        tuple: (yolo_results_dict, annotated_frame)
    """
    results = yolo_model(frame, verbose=False)
    annotated = results[0].plot()

    yolo_results = {}
    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            if cls_name not in yolo_results or conf > yolo_results[cls_name]["conf"]:
                yolo_results[cls_name] = {
                    "bbox": [int(b) for b in bbox],
                    "conf": round(conf, 4)
                }

    return yolo_results, annotated


def get_action(ppo_agent, obs):
    """Get action from PPO agent or rule-based fallback.

    Args:
        ppo_agent: Trained PPO model, or None.
        obs (np.ndarray): 10-dim observation vector.

    Returns:
        int: Action (0, 1, or 2).
    """
    if ppo_agent is not None:
        action, _ = ppo_agent.predict(obs, deterministic=True)
        return int(action)

    # Rule-based fallback (same thresholds as reward function)
    phone_dur = obs[5]
    gaze_dur = obs[6]
    if phone_dur > 3.0 or gaze_dur > 4.0:
        return 2
    elif phone_dur > 1.0 or gaze_dur > 1.5:
        return 1
    return 0


def draw_hud(frame, action, attention_score, frame_count, violation_count,
             phone_dur, gaze_dur, cig_dur):
    """Draw the heads-up display overlay on the frame.

    Renders action label (top-left), attention score (top-right),
    and a bottom info bar with stats.
    """
    h, w = frame.shape[:2]
    cfg = ACTION_CONFIG.get(action, ACTION_CONFIG[0])

    # ---- Top-left: Action label ----
    cv2.putText(frame, cfg["label"], (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, cfg["color"], 2, cv2.LINE_AA)

    # ---- Top-right: Attention Score ----
    score_color = ((0, 255, 0) if attention_score < 40 else
                   (0, 200, 255) if attention_score < 70 else
                   (0, 0, 255))
    score_txt = f"Score: {attention_score:.0f}/100"
    ts = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, score_txt, (w - ts[0] - 12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)

    # ---- Bottom info bar (semi-transparent) ----
    bar_h = 45
    bar_top = h - bar_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    info = (f"Frame: {frame_count}  |  Violations: {violation_count}  |  "
            f"Phone: {phone_dur:.1f}s  |  Gaze: {gaze_dur:.1f}s  |  "
            f"Cig: {cig_dur:.1f}s")
    cv2.putText(frame, info, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ==========================================================================
# Main Entry Point
# ==========================================================================

def main(video_source=0):
    """Run the full SmartRoad AI integrated pipeline (FROZEN).

    Args:
        video_source: 0 for webcam, or path to .mp4 video file.
    """
    print("=" * 65)
    print(f"  SmartRoad AI — Final Integrated Pipeline v{VERSION}")
    print(f"  IEEE CIS VIT Pune  |  ML2308 Artificial Intelligence")
    print("=" * 65)

    # ---- Load everything ----
    models = load_all_models()
    logger = ViolationLogger(csv_path="violations.csv", evidence_dir="evidence")

    # ---- Open video source ----
    print(f"[VIDEO] Opening source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_source}")
        sys.exit(1)

    # ---- State ----
    frame_count = 0
    seg_results = {"driver_zone": False, "steering_visible": True}
    seg_overlay = None
    reset_tracker()

    print("[RUN] Pipeline running. Press 'q' to quit, 's' for screenshot.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(video_source, str):
                    print("[INFO] End of video file.")
                    break
                continue

            frame_count += 1

            # ---- Layer 1: SegFormer ----
            if frame_count % SEG_INTERVAL == 0 or seg_overlay is None:
                seg_results, seg_overlay = run_segformer(
                    frame, models["seg_processor"], models["seg_model"]
                )

            # ---- Layer 2: YOLOv8 ----
            yolo_results, annotated = run_yolo(frame, models["yolo"])

            # ---- Build observation ----
            obs = build_observation(yolo_results, seg_results)

            # ---- Layer 3: RL Agent ----
            action = get_action(models["ppo"], obs)

            # ---- Attention Score ----
            ts = get_tracker_state()
            attention = compute_attention_score_from_durations(
                ts["phone_duration"], ts["gaze_duration"],
                ts.get("cigarette_duration", 0.0)
            )

            # ---- Log violations ----
            if action == 2:
                now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
                logger.log_violation(
                    action=action, obs_vector=obs,
                    timestamp=now, frame=frame,
                    attention_score=attention
                )

            # ---- Compose display ----
            display = (cv2.addWeighted(annotated, 0.6, seg_overlay, 0.4, 0)
                       if seg_overlay is not None else annotated.copy())

            display = draw_hud(
                display, action, attention, frame_count,
                logger.violation_count,
                ts["phone_duration"], ts["gaze_duration"],
                ts.get("cigarette_duration", 0.0)
            )

            # ---- Display ----
            try:
                cv2.imshow("SmartRoad AI v2.0", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"screenshot_{frame_count}.jpg", display)
            except cv2.error:
                pass  # Headless environment

            # ---- Console logging ----
            if frame_count % FPS == 0:
                lbl = ACTION_CONFIG.get(action, ACTION_CONFIG[0])["label"]
                print(f"  [{frame_count:>6d}] {lbl:<12s} "
                      f"Score: {attention:>5.1f}  "
                      f"Ph: {ts['phone_duration']:.1f}s  "
                      f"Gz: {ts['gaze_duration']:.1f}s")

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = logger.get_session_summary()
        print(f"\n{'=' * 65}")
        print(f"  FINAL SESSION SUMMARY")
        print(f"  Frames: {frame_count}  |  "
              f"Violations: {summary['total_violations']}  |  "
              f"Log: {summary['csv_path']}")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg
    main(video_source=source)
