# RL Module Documentation — SmartRoad AI

## 1. Overview

The RL module consists of two core components that enable reinforcement learning for driver distraction detection:

- **`obs_builder.py`**: Transforms raw pipeline outputs (YOLO object detections + SegFormer semantic segmentation) into structured 10-dimensional observation vectors. Includes a `DurationTracker` that monitors how long various distractions persist across frames.

- **`rl_environment.py`**: Implements a Gymnasium-compatible custom environment (`DriverEnv`) where an agent observes driver behavior and chooses one of three actions: `ALL_CLEAR`, `MONITOR`, or `VIOLATION`. The environment provides rewards based on action correctness and terminates episodes when safety thresholds are exceeded.

These modules integrate with the live webcam pipeline from `pipeline.py`, which runs SegFormer-b0 for semantic segmentation and YOLOv8 Nano for object detection. The RL agent learns to classify distraction severity in real-time, balancing false positives against missed violations.

---

## 2. Observation Space

The observation space is a 10-dimensional continuous vector with the following structure:

| Index | Name                   | Range      | Source     | Description                                                  |
|-------|------------------------|------------|------------|--------------------------------------------------------------|
| 0     | `phone_detected`       | [0.0, 1.0] | YOLO       | Binary indicator: 1.0 if cell phone detected, 0.0 otherwise |
| 1     | `phone_conf`           | [0.0, 1.0] | YOLO       | Confidence score of phone detection (0.0 if not detected)   |
| 2     | `phone_near_face`      | [0.0, 1.0] | YOLO       | Binary heuristic: 1.0 if phone bbox y1 < 192 (upper 40%)    |
| 3     | `gaze_away`            | [0.0, 1.0] | SegFormer  | Binary indicator: 1.0 if steering wheel not visible         |
| 4     | `cigarette_detected`   | [0.0, 1.0] | YOLO       | Binary indicator: 1.0 if cigarette detected                 |
| 5     | `phone_duration`       | [0.0, 30.0]| Tracker    | Duration (seconds) phone has been present, capped at 30.0   |
| 6     | `gaze_duration`        | [0.0, 30.0]| Tracker    | Duration (seconds) gaze has been away, capped at 30.0       |
| 7     | `cigarette_duration`   | [0.0, 30.0]| Tracker    | Duration (seconds) cigarette has been present, capped at 30.0|
| 8     | `person_detected`      | [0.0, 1.0] | YOLO       | Binary indicator: 1.0 if person detected in frame           |
| 9     | `driver_zone_occupied` | [0.0, 1.0] | SegFormer  | Binary indicator: 1.0 if driver zone segmentation is active |

**Note**: Duration values use gradual decay (subtract 2-3 frames when distraction disappears) instead of instant reset. This prevents brief occlusions from resetting counters to zero.

---

## 3. Action Space

The action space is discrete with 3 possible actions:

| Action Index | Label       | Meaning                                                  | When to Choose                                      |
|--------------|-------------|----------------------------------------------------------|-----------------------------------------------------|
| 0            | ALL_CLEAR   | No distraction detected, driver is attentive             | phone_duration ≤ 1.0s AND gaze_duration ≤ 1.5s     |
| 1            | MONITOR     | Minor distraction observed, needs watching               | 1.0s < phone_duration ≤ 3.0s OR 1.5s < gaze ≤ 4.0s |
| 2            | VIOLATION   | Serious distraction detected, flag for intervention      | phone_duration > 3.0s OR gaze_duration > 4.0s      |

The agent's goal is to learn the correct action for each observation, minimizing false positives while catching all real violations.

---

## 4. Reward Function

The reward function balances three priorities:
1. Catch serious violations (phone > 3s, gaze > 4s)
2. Avoid false positives (unnecessary violation flags)
3. Use MONITOR for borderline cases instead of jumping to VIOLATION

| Condition                                          | Reward | Reasoning                                                  |
|----------------------------------------------------|--------|------------------------------------------------------------|
| Action = VIOLATION, actual violation exists        | +10.0  | Correct flag of serious distraction (highest reward)       |
| Action = VIOLATION, no actual violation            | -5.0   | False positive — wastes operator attention                 |
| Action = MONITOR, actual violation exists          | -2.0   | Too cautious — should have flagged immediately             |
| Action = MONITOR, minor distraction (1-3s phone)   | +2.0   | Correct monitoring of developing situation                 |
| Action = MONITOR, no distraction                   | -1.0   | Unnecessary monitoring when driver is fine                 |
| Action = ALL_CLEAR, actual violation exists        | -8.0   | Missed serious violation — very dangerous                  |
| Action = ALL_CLEAR, minor distraction              | -1.0   | Should be monitoring but said all clear                    |
| Action = ALL_CLEAR, no distraction                 | +0.5   | Correct all clear — prevents over-flagging                 |

**Key Design Choice**: The small positive reward (+0.5) for correct ALL_CLEAR prevents the agent from flagging violations too aggressively. Without this, the agent might always choose VIOLATION to avoid the -8.0 penalty.

---

## 5. Episode Termination

An episode ends (`terminated=True`) when **any** of the following conditions are met:

1. **Max Steps Reached**: `step_count >= max_steps` (default 200)
   - Standard episode length limit

2. **Too Many Violations**: `violation_count >= max_violations_per_episode` (default 10)
   - Counts false positives and missed violations
   - Ends episode early if agent makes too many mistakes

3. **Extreme Distraction** (NEW): `phone_duration > 25.0` OR `gaze_duration > 25.0`
   - Safety cutoff — if distraction persists this long, episode is considered failed
   - Prevents runaway episodes where agent ignores severe violations

This three-way termination logic ensures episodes don't run indefinitely and encourages the agent to flag violations before they reach critical levels.

---

## 6. How to Run

### Basic Usage

```python
from rl_environment import DriverEnv
from obs_builder import reset_tracker

# Create environment (uses FakePipeline for testing)
env = DriverEnv(pipeline_fn=None, max_steps=200)

# Reset environment
obs, info = env.reset(seed=42)

# Run episode
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        print(f"Episode ended at step {step}")
        print(f"Stats: {env.episode_stats()}")
        break
```

### Integration with Live Pipeline

```python
from rl_environment import DriverEnv
from pipeline import get_frame_results  # Assumes pipeline.py exposes this

# Use live webcam pipeline
env = DriverEnv(pipeline_fn=get_frame_results, max_steps=500)

obs, info = env.reset()
while True:
    action = agent.predict(obs)  # Use trained PPO agent
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # Display frame with action overlay

    if terminated:
        obs, info = env.reset()
```

### Validation

```python
from gymnasium.utils.env_checker import check_env

env = DriverEnv(pipeline_fn=None)
check_env(env.unwrapped)  # Validates Gymnasium compliance
```

---

## 7. Integration with PPO

The `train_ppo.py` script (owned by another team member) should import and use `DriverEnv` as follows:

```python
from stable_baselines3 import PPO
from rl_environment import DriverEnv

# Create vectorized environment
env = DriverEnv(pipeline_fn=None, max_steps=300)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100000)

# Save model
model.save("smartroad_ppo_agent")
```

The PPO agent will learn to:
- Map observations → actions through a neural network policy
- Optimize for cumulative reward over episodes
- Balance exploration (trying new actions) vs exploitation (using known good actions)

After training, the agent can be deployed in the live pipeline to make real-time distraction classifications.

---

## 8. Design Decisions

### Key Non-Obvious Choices

1. **Gradual Decay Instead of Instant Reset**
   - When a distraction disappears, the duration tracker decreases by 2-3 frames per step instead of resetting to zero immediately.
   - **Why**: Handles brief occlusions (e.g., hand momentarily blocking phone) without losing track of ongoing distractions.
   - **Trade-off**: May slightly overestimate duration if distraction truly ends, but prevents noisy detections from breaking duration tracking.

2. **Phone Near Face Heuristic (y1 < 192)**
   - Uses bbox y-coordinate to guess if phone is near driver's face vs in lap.
   - **Why**: Phone near face is more distracting than phone in lap (visual + manual distraction).
   - **Limitation**: Simple heuristic, not perfect — future work could use pose estimation.

3. **Small Positive Reward for Correct ALL_CLEAR (+0.5)**
   - Agent gets a small reward for correctly saying "all clear" when no distraction exists.
   - **Why**: Without this, agent has no incentive to ever use ALL_CLEAR — it would always flag violations to avoid the large -8.0 penalty for missed violations.
   - **Impact**: Encourages the agent to be selective and only flag real violations.

4. **Asymmetric Penalties (Missed Violation -8.0 vs False Positive -5.0)**
   - Missing a violation is penalized more harshly than a false positive.
   - **Why**: In safety-critical applications, failing to catch a distracted driver is worse than annoying an operator with a false alarm.
   - **Context**: Aligns with real-world priorities where recall > precision for safety violations.

5. **FakePipeline Uses Seeded RNG from Gymnasium**
   - The synthetic data generator uses `self.np_random` from Gymnasium's base class.
   - **Why**: Ensures reproducibility when `env.reset(seed=X)` is called — critical for debugging and benchmarking.
   - **Implementation**: Handles both `numpy.random.Generator` (modern) and `RandomState` (legacy) for compatibility.

---

## Summary

This RL module provides a complete Gymnasium environment for training agents to detect driver distractions. The observation space captures both instant detections (YOLO, SegFormer) and temporal patterns (duration tracking), while the reward function teaches the agent to balance precision and recall. The environment integrates seamlessly with the existing SmartRoad AI pipeline and is ready for PPO training.
