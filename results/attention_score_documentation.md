# SmartRoad AI: Attention Score Documentation

## 1. Metric Overview

The **Attention Score** is a novel, continuous metric developed for SmartRoad AI to quantify driver distraction severity in real-time. Rather than reacting purely to single-frame detections (which are prone to noise and occlusions), the Attention Score aggregates multi-class distraction durations into a unified 0–100 scale.

- **0** = Highly attentive, fully focused on the road.
- **100** = Maximum cognitive and manual distraction.

---

## 2. The Formula

The score is calculated at every frame step using the following linear combination of duration timers, which is then normalized:

```python
AttentionScore = (
    phone_duration * 0.4     # Weight 1: Highest danger
  + gaze_duration * 0.3      # Weight 2: Severe danger 
  + cigarette_duration * 0.2 # Weight 3: Moderate danger
  + activity_count * 0.1     # Weight 4: Compounding multiplier
) 

Normalized_Score = (AttentionScore / 28.0) * 100.0
```

*Note: All durations are capped at 30.0 seconds to match the observation vector ceiling constraint.*

### 2.1 Weighting Rationale
1. **Phone Usage (0.4)**: Assigned the highest weight because mobile phone usage combines manual, visual, and cognitive distraction. It is the leading cause of preventable accidents.
2. **Gaze Away (0.3)**: Looking away from the road for extended periods is highly dangerous. We assign this the second highest weight. A driver checking a mirror is brief, so the duration threshold prevents immediate spikes.
3. **Cigarette / Smoking (0.2)**: Smoking is primarily a manual distraction, taking one hand off the wheel. It is dangerous, but statistically less fatal than texting, hence the lower weight.
4. **Activity Count (0.1)**: Represents the number of simultaneous distractions (e.g., smoking *while* looking at a phone). This provides a flat bump to the score when multiple bad behaviors overlap.

---

## 3. Gradual Decay vs. Instant Reset

A common problem in object detection pipelines is flickering (a phone is detected in frame 1, lost in frame 2 due to blur, and found again in frame 3). If a timer resets instantly to 0 when an object is lost, the agent might never catch a violation.

Instead of an instant reset, `DurationTracker` implements **gradual decay**:
```python
# Instead of: self.phone_frames = 0
self.phone_frames = max(0, self.phone_frames - 3)
```
This means if a driver is texting for 10 seconds, and their hand briefly obscures the phone for 5 frames, the timer only drops by a small faction (e.g. down to 9.5s) instead of resetting to 0. If they put the phone away permanently, the score smoothly decays back to zero over a couple of seconds.

---

## 4. Calibration & Action Thresholds

During daylight testing across 6 mock scenarios, we mapped the continuous Attention Score to the three discrete Reinforcement Learning actions (0, 1, 2) to ensure the PPO agent's logic aligned with human intuition.

| Attention Score | System Action | Description / Meaning |
|-----------------|--------------|-----------------------|
| `0.0 — 59.9` | **ALL CLEAR (0)** | Driver is focused. Brief glances away (checking mirrors) or scratching nose are ignored. |
| `60.0 — 79.9` | **MONITOR (1)** | Suspicious continuous behavior. For example, looking away for > 1.5 seconds or brief phone usage. System is on high alert. |
| `80.0 — 100.0` | **VIOLATION (2)** | Confirmed severe infraction. Phone use > 3.0s or gaze away > 4.0s. Triggers the logging and e-challan evidence capture. |

*(See `attention_score_plot.png` in the `/results/` directory for a visual representation of how the score moves through these threshold zones during a driving scenario).*
