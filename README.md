# SmartRoad AI 

An intelligent, context-aware traffic safety enforcement system designed to tackle the critical issue of distracted driving. Road safety enforcement traditionally relies on single-frame speed cameras which are incapable of identifying sustained dangerous behaviors like texting, looking away from the road, or drowsiness. SmartRoad AI addresses this gap by employing a robust three-layer artificial intelligence pipeline. 

First, a SegFormer semantic segmentation model analyzes the spatial layout of the vehicle cabin, identifying zones like the driver's seat and the steering wheel. Second, a YOLOv8 object detection model operates in real-time to pinpoint the exact locations of phones, cigarettes, faces, and seatbelts. Finally, we introduce a novel Attention Score metric fed into a Reinforcement Learning agent (PPO). Instead of raising an alert on a single anomalous frame—which often leads to false positives—the RL agent tracks sustained behavioral changes and makes a contextual, highly reliable decision to flag a genuine violation.

Once a prolonged violation is confirmed, our automated extraction pipeline logs the event, saves the evidence frame, and sets the foundation for an automatic E-Challan generation system linked to traffic authority databases. This project demonstrates state-of-the-art applied Computer Vision and RL merging for proactive road safety.

---

## Table of Contents

| Section | Reference |
|---------|-----------|
| [What This Project Delivers](#what-this-project-delivers) | Capabilities and output overview |
| [Project Layers](#project-layers) | Three-layer system breakdown |
| &emsp;[Layer 1 — Semantic Segmentation](#layer-1--semantic-segmentation) | SegFormer scene understanding |
| &emsp;[Layer 2 — Object Detection](#layer-2--object-detection) | YOLOv8 real-time detection |
| &emsp;[Layer 3 — RL Agent](#layer-3--rl-agent) | PPO intelligent decision-making |
| [How the Layers Work Together](#how-the-layers-work-together) | Combined pipeline flow |
| [Automatic Challan Generation](#automatic-challan-generation) | End-to-end enforcement pipeline |
| [Tech Stack](#tech-stack) | Technologies and tools used |
| [Future Scope](#future-scope) | Planned enhancements |

---

## What This Project Delivers

SmartRoad AI is a **three-layer AI pipeline** that runs on a dashboard or traffic camera and delivers:

- **Scene Understanding** — Every pixel in the frame is labeled: driver seat, steering wheel, dashboard, road view, driver body. The system knows the spatial layout of the vehicle environment.
- **Real-Time Object Detection** — YOLOv8 identifies phones, cigarettes, faces, and seatbelts frame-by-frame with bounding boxes and confidence scores.
- **Intelligent Violation Decisions** — A Reinforcement Learning agent reads scene context and decides: `All Clear`, `Monitor`, or `Violation` — based on sustained behavior, not a single frame.
- **Automatic E-Challan Generation** — Once a violation is confirmed, the system captures evidence, reads the license plate via ANPR, logs the violation, and generates a challan automatically.

---

## Project Layers

### Layer 1 — Semantic Segmentation

**Model:** `nvidia/segformer-b0-finetuned-ade-512-512` (HuggingFace)

**What it does:**
Semantic segmentation assigns a label to every single pixel in the camera frame. Unlike object detection which draws boxes, segmentation understands the *structure* of the entire driving environment.

**What it labels:**
- `driver seat` — driver position zone
- `steering wheel area` — hands-on-wheel zone
- `dashboard` — instrument panel region
- `road view area` — forward visibility zone
- `driver body position` — posture and orientation

**Why it matters:**
This is the spatial context layer. Without it the RL agent cannot know if a phone is near the driver's face or just sitting on the seat. Segmentation provides that understanding.

**Output:** A color-coded overlay on the camera feed where each zone is a distinct color.

---

### Layer 2 — Object Detection

**Model:** `YOLOv8 Nano` (Ultralytics)

**What it does:**
YOLO scans every frame and draws bounding boxes around detected objects with class labels and confidence scores.

**What it detects:**
- Mobile phone, cigarette, cup — distraction objects
- Driver face — for gaze and drowsiness tracking
- Steering wheel — hands-on-wheel verification
- Seatbelt — compliance detection

**Why it matters:**
Segmentation gives zone context; YOLO gives object-level precision. Together they answer: *"A phone is detected (YOLO) and it is positioned near the driver's face zone (Segmentation) for more than 3 seconds."*

**Output:** Bounding boxes with labels like `phone 0.91`, `cigarette 0.82` on the live feed.

---

### Layer 3 — RL Agent

**Algorithm:** `PPO` — Proximal Policy Optimization (Stable-Baselines3)

**What it does:**
The RL agent is a trained decision-maker. It takes the combined output of Layer 1 and Layer 2 as its **observation** and decides whether a traffic violation has occurred.

**Observation space (what it sees):**
- Semantic zone labels from SegFormer
- Detected objects from YOLOv8
- Driver attention duration
- Frequency of unsafe actions
- Time-based activity tracking

**Action space (what it decides):**

| Action | Trigger Condition |
|--------|------------------|
| `0 — All Clear` | Driver focused, hands on wheel, no distractions |
| `1 — Monitor` | Suspicious activity detected, observing pattern |
| `2 — Violation` | Confirmed unsafe behavior exceeding time threshold |

**Reward function:**
- `+1` for correctly detecting a violation
- `-1` for a false alert
- `-2` for missing a real violation

---

## How the Layers Work Together

```
Camera Frame (Dashboard / Traffic Camera)
     │
     ├──► Layer 1: SegFormer ──► Scene Zone Map (seat / steering / dashboard / road)
     │
     ├──► Layer 2: YOLOv8   ──► Object Detections (phone / cigarette / face / seatbelt)
     │
     └──► Layer 3: RL Agent ──► reads Zone Map + Detections ──► All Clear / Monitor / Violation
                                                                          │
                                                              Challan Generation Pipeline
```

Segmentation runs every 10 frames (CPU-friendly). YOLO runs every frame. The RL agent combines both and makes a violation decision in real-time.

---

## Automatic Challan Generation

Once the RL agent confirms a violation, the system triggers a five-step enforcement pipeline:

| Step | Action |
|------|--------|
| 1 — Capture Evidence | Violation frame saved with timestamp |
| 2 — License Plate Detection | ANPR module detects vehicle number plate |
| 3 — OCR Extraction | EasyOCR reads the plate number from the image |
| 4 — Violation Logging | Vehicle number, violation type, timestamp, location, evidence stored |
| 5 — E-Challan Generation | Challan automatically generated and sent to traffic authority database |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 Nano — Ultralytics |
| Semantic Segmentation | SegFormer-b0 — HuggingFace Transformers |
| Face + Pose Tracking | MediaPipe |
| RL Agent | PPO — Stable-Baselines3 |
| RL Environment | Gymnasium |
| License Plate OCR | EasyOCR |
| Camera + Video | OpenCV |
| Dashboard UI | Streamlit |
| Deep Learning | PyTorch |

## Future Scope

- Multi-lane roadside camera support for large-scale enforcement
- Night vision and low-light model fine-tuning
- Real-time SMS alert to traffic authority on violation
- Integration with national vehicle registration database
- Edge deployment on Raspberry Pi / Jetson Nano for in-vehicle use
- Attention Score tracking — rolling distraction percentage over a 10-minute window

---

## Module Overview

| Component | File | Description |
|-----------|------|-------------|
| Environment | `rl_environment.py` | Custom Gymnasium environment defining state/action/rewards |
| State Vector | `obs_builder.py` | Reduces high-dimensional visual output into the 10-D observation state |
| Logic Pipeline | `pipeline.py` | Core handler for SegFormer and YOLOv8 real-time inference |
| Enforcement | `alert_logger.py` | Triggers violation logs and saves confirmation image evidence |
| Engine | `final_integrate.py` | Final end-to-end event execution loop merging all layers |

## How to Run

1. Clone the repository and navigate or open a terminal in the folder.
2. Create and activate a virtual environment (e.g. `python -m venv venv` and `venv\Scripts\activate` on Windows).
3. Install dependencies: `pip install -r requirements.txt`
4. Run the final integrated dashboard: `python final_integrate.py`

## Team Framework
| Member | Role | Key Contribution |
|--------|------|------------------|
| **Sachi** | Pipeline Lead | End-to-end `final_integrate.py` tying Seg+YOLO+RL |
| **Ragini** | RL Env Lead | Gymnasium `rl_environment.py` and `obs_builder.py` |
| **Nikhil** | RL Agent Lead | PPO model training and hyperparameters |
| **Shakti** | Metrics Lead | Precision/Recall statistics and `attention_score.py` metrics |
| **Sanat** | Data & Docs | Project roadmap consolidation, scenario sets, and artifact summaries |
