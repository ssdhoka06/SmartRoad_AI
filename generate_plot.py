"""Generate Attention Score Plot for SmartRoad AI.

Simulates a 2-minute scenario with different distraction events, calculates the continuous
Attention Score, and plots the timeline using matplotlib.

Author: Shakti (Attention Score & Metrics)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from attention_score import DurationTracker, compute_attention_score

def generate_plot():
    tracker = DurationTracker()
    fps = 30
    total_seconds = 120
    total_frames = total_seconds * fps
    
    scores = []
    times = []
    activities = []
    
    for frame in range(total_frames):
        t = frame / fps
        times.append(t)
        
        # Scenario timeline:
        # 0s - 20s: Attentive driving
        # 20s - 35s: Phone use while looking ahead
        # 35s - 50s: Phone drops (occlusion decay), still looking ahead
        # 50s - 65s: Gaze away (checking mirror/outside too long)
        # 65s - 90s: Attentive driving
        # 90s - 110s: Phone use AND Gaze away simultaneously (high danger)
        # 110s - 120s: Recovery
        
        detections = {"phone": False, "gaze_away": False, "cigarette": False}
        activity_count = 0
        
        if 20 <= t < 35:
            detections["phone"] = True
            activity_count = 1
        elif 50 <= t < 65:
            detections["gaze_away"] = True
            activity_count = 1
        elif 90 <= t < 110:
            detections["phone"] = True
            detections["gaze_away"] = True
            activity_count = 2
            
        tracker.update(detections, fps=fps)
        score = compute_attention_score(tracker, activity_count=activity_count)
        scores.append(score)
        
        if score >= 80:
            activities.append('VIOLATION')
        elif score >= 60:
            activities.append('MONITOR')
        else:
            activities.append('ALL_CLEAR')

    # Plot figure
    plt.figure(figsize=(12, 6))
    
    # Draw zones
    plt.axhspan(0, 60, color='lightgreen', alpha=0.3, label='Safe Zone (<60)')
    plt.axhspan(60, 80, color='navajowhite', alpha=0.4, label='Monitor Zone (60-80)')
    plt.axhspan(80, 100, color='lightcoral', alpha=0.3, label='Violation Zone (>80)')
    
    # Plot score line
    plt.plot(times, scores, color='darkblue', linewidth=2.5, label='Attention Score')
    
    # Event annotations
    plt.annotate('Starts Phone', xy=(20, 0), xytext=(20, 20), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Stops Phone', xy=(35, scores[35*fps]), xytext=(35, 90), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Gaze Away', xy=(50, 0), xytext=(45, 30), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Phone + Gaze Away', xy=(90, 0), xytext=(85, 40), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('SmartRoad AI: Attention Score Timeline (Simulated Scenario)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Attention Score (Max = 100)', fontsize=12)
    plt.xlim(0, 120)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    out_dir = r'c:\Users\shakt\OneDrive\Desktop\SmartRoad_AI\results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'attention_score_plot.png')
    
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    generate_plot()
