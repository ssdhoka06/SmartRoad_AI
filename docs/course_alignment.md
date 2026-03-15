# Course Alignment: ML2308 Artificial Intelligence

This document details how the SmartRoad AI project directly aligns with the core AI concepts explored in the ML2308 course syllabus.

## 1. Intelligent Agents
- **Course Concept:** Rational Agents, Agent environments mapping Percepts to Actions.
- **SmartRoad AI Component:** The Proximal Policy Optimization (PPO) agent running the `DriverEnv`. The system operates continuously, sensing its vehicle environment (percepts) via the camera and deciding whether to output an alert/violation (actions) autonomously.

## 2. State Space and Observation Formulations
- **Course Concept:** Structuring real-world environments into constrained numeric states for machine solving.
- **SmartRoad AI Component:** `obs_builder.py` reduces highly unstructured high-dimensional input (1080p RGB frames) into a tightly bounded 10-dimensional state vector composed of logical constraints (e.g., "Phone near face", "Phone duration"). 

## 3. Machine Learning (Supervised Learning)
- **Course Concept:** Deep learning, Computer Vision applications.
- **SmartRoad AI Component:** Object detection (YOLOv8) and Semantic scene segmentation (SegFormer-b0). These represent state-of-the-art applied convolutional / transformer-based supervised visual processing applied to spatial awareness.

## 4. Rational Decision Making and Reinforcement Learning
- **Course Concept:** Utility Functions, MDP (Markov Decision Processes), Reward formulation.
- **SmartRoad AI Component:** The environment provides rewards (+1 for correct violation detection, -1 for false alarm, -2 for missed violation). Over 50,000 timesteps of training, the agent optimizes its policy to maximize its long-term sequence utility.

## 5. Production Expert Systems
- **Course Concept:** Translating AI heuristics into production logging boundaries.
- **SmartRoad AI Component:** `final_integrate.py` and `alert_logger.py` build upon the AI logic to bridge to deterministic logging, representing inference-time deployment bounds necessary for generating legal E-Challans.
