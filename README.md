# CampusWatch 

An intelligent, context-aware campus safety monitor that goes beyond object detection it understands scenes, recognizes behavioral patterns, and makes real-time alert decisions using Reinforcement Learning.

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
| [Tech Stack](#tech-stack) | Technologies and tools used |
| [Future Scope](#future-scope) | Planned enhancements |

---

## What This Project Delivers

VIT CampusWatch is a **three-layer AI pipeline** that runs live on a webcam and delivers:

- **Scene Understanding** — Every pixel in the frame is labeled: person, desk, door, floor, wall. The system knows *where* things are, not just *what* they are.
- **Real-Time Object Detection** — YOLOv8 identifies and tracks objects frame-by-frame with bounding boxes and confidence scores.
- **Intelligent Alert Decisions** — A Reinforcement Learning agent reads the scene context and decides: `All Clear`, `Monitor`, or `Alert` — based on *behavior*, not just presence.
- **Context-Aware Safety** — A person sitting at a desk is normal. A person loitering near an exit is not. CampusWatch knows the difference.

## Project Layers

### Layer 1 — Semantic Segmentation

**Model:** `nvidia/segformer-b0-finetuned-ade-512-512` (HuggingFace)

**What it does:**
Semantic segmentation assigns a label to every single pixel in the webcam frame. Unlike object detection which draws boxes, segmentation understands the *structure* of the entire scene.

**What it labels:**
- `person` — any human in the frame
- `desk / table` — workspace zones
- `door` — entry/exit points
- `floor / wall / ceiling` — spatial boundaries
- 147 other ADE20K categories

**Why it matters:**
This is the "eyes" of the system. Without scene understanding, the RL agent cannot know if a person is in a safe zone or a restricted zone. Segmentation provides that spatial context.

**Output:** A color-coded overlay on the webcam feed where each zone is a distinct color.

---

### Layer 2 — Object Detection

**Model:** `YOLOv8 Nano` (Ultralytics)

**What it does:**
YOLO (You Only Look Once) scans every frame and draws bounding boxes around detected objects with class labels and confidence scores.

**What it detects:**
- People, chairs, desks, bags, laptops, phones — 80 COCO classes total

**Why it matters:**
Segmentation gives zone context; YOLO gives object-level precision. Together they answer: *"There is a person (YOLO), and they are standing near a door zone (Segmentation)."*

**Output:** Bounding boxes with labels like `person 0.90`, `backpack 0.74` on the live feed.

---

### Layer 3 — RL Agent

**Algorithm:** `PPO` — Proximal Policy Optimization (Stable-Baselines3)

**What it does:**
The RL agent is a trained decision-maker. It takes the combined output of Layer 1 and Layer 2 as its **observation** and decides what action to take.

**Observation space (what it sees):**
- Semantic zone labels in the frame
- Number of persons detected
- Proximity of persons to exit/door zones
- Time-in-zone counter

**Action space (what it decides):**

| Action | Trigger Condition |
|--------|------------------|
| `0 — All Clear` | Person in desk/work zone, normal behavior |
| `1 — Monitor` | Person near exit, idle for extended time |
| `2 — Alert` | Person in restricted zone or suspicious pattern |

**Reward function:**
- `+1` for correct alert
- `-1` for false alert
- `-2` for missed threat

**Why it matters:**
This is what makes CampusWatch different from plain YOLO. Any camera can detect a person. Only an intelligent agent can decide whether that person's *context* is a threat.

---

## How the Layers Work Together

```
Webcam Frame
     │
     ├──► Layer 1: SegFormer ──► Scene Zone Map (person / desk / door / floor)
     │
     ├──► Layer 2: YOLOv8   ──► Object Bounding Boxes + Labels
     │
     └──► Layer 3: RL Agent ──► reads Zone Map + Boxes ──► All Clear / Monitor / Alert
                                                                    │
                                                            Streamlit Dashboard
```

Segmentation runs every 10 frames (CPU-friendly). YOLO runs every frame (lightweight). The RL agent reads both outputs and decides the safety status in real-time.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 Nano — Ultralytics |
| Semantic Segmentation | SegFormer-b0 — HuggingFace Transformers |
| RL Agent | PPO — Stable-Baselines3 |
| RL Environment | Gymnasium |
| Webcam + Video | OpenCV |
| Dashboard UI | Streamlit |
| Deep Learning | PyTorch |


## Future Scope

- Multi-camera support for full campus coverage
- Night vision / low-light model fine-tuning
- Alert notification system via email/SMS
- Historical incident logging and analytics dashboard
- Edge deployment on Raspberry Pi / Jetson Nano
