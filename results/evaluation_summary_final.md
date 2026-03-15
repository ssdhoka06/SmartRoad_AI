# SmartRoad AI — Final Scenario Evaluation Summary

## Overall Performance
- **Accuracy:** 100.00%
- **Total Frames Evaluated:** 30000
- **Avg Attention Score:** 0.32
- **Max Attention Score:** 1.49

## Per-Class Breakdown

| Action Label | Precision | Recall | F1-Score | Support (Frames) |
|--------------|-----------|--------|----------|------------------|
| ALL_CLEAR (0) | 1.00 | 1.00 | 1.00 | 30000.0 |
| MONITOR (1) | 0.00 | 0.00 | 0.00 | 0.0 |
| VIOLATION (2) | 0.00 | 0.00 | 0.00 | 0.0 |

## Confusion Matrix
*(Rows = Expected Actual, Columns = Agent Prediction)*

| | Predicted Clear | Predicted Monitor | Predicted Violation |
|---|---|---|---|
| **Actual Clear** | 30000 | 0 | 0 |
| **Actual Monitor** | 0 | 0 | 0 |
| **Actual Violation** | 0 | 0 | 0 |

---
*Note: Evaluation generated dynamically by mapping observation vectors to ground truth state rules, covering 100 PPO episodes.*