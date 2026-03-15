"""Generate Final Metrics Report for SmartRoad AI.

Reads Nikhil's final_eval.csv agent evaluation file and applies pseudo-ground-truth
logic to evaluate precision, recall, and F1 score since test videos are pending.

Author: Shakti (Attention Score & Metrics)
"""

import os
import csv
import ast
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def generate_metrics():
    # 1. Read Nikhil's Final Eval
    file_path = r'c:\Users\shakt\OneDrive\Desktop\SmartRoad_AI\final_eval.csv'
    
    predictions = []
    ground_truths = []
    attention_scores = []
    
    # Generate an "expected" / ground truth action based on the observation vector constraints
    # Observation vector format (from obs_builder):
    # [phone_det, phone_conf, phone_near_face, gaze_away, cig_det,
    #  phone_dur, gaze_dur, cig_dur, person_det, driver_zone]
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header (usually Episode, Step, Reward, Action, Observation)
        
        for row in reader:
            try:
                # Based on final_eval.csv column structure: 
                # Episode[0], Step[1], Action[2], Reward[3], Obs[4]
                action = int(row[2])
                obs_str = row[4].replace('[', '').replace(']', '').split(',')
                obs = [float(x.strip()) for x in obs_str]
                
                phone_dur = obs[5]
                gaze_dur = obs[6]
                cig_dur = obs[7]
                
                # Re-calculate Attention Score
                activity_count = int(obs[0] + obs[3] + obs[4])
                score = (phone_dur * 0.4 + gaze_dur * 0.3 + cig_dur * 0.2 + activity_count * 0.1) / 28.0 * 100
                score = min(score, 100.0)
                attention_scores.append(score)
                
                # Calculate Ground Truth based on explicit business rules from roadmap:
                if phone_dur > 3.0 or gaze_dur > 4.0:
                    expected = 2  # VIOLATION
                elif phone_dur > 1.0 or gaze_dur > 1.5:
                    expected = 1  # MONITOR
                else:
                    expected = 0  # ALL CLEAR
                    
                ground_truths.append(expected)
                predictions.append(action)
                
            except Exception as e:
                pass


    # 2. Compute Metrics using sklearn
    target_names = ["ALL_CLEAR (0)", "MONITOR (1)", "VIOLATION (2)"]
    labels = [0, 1, 2]
    
    # Full report
    report = classification_report(ground_truths, predictions, labels=labels, target_names=target_names, zero_division=0)
    acc = accuracy_score(ground_truths, predictions)
    cm = confusion_matrix(ground_truths, predictions, labels=labels)
    
    # 3. Write Metrics Report txt
    out_dir = r'c:\Users\shakt\OneDrive\Desktop\SmartRoad_AI\results'
    os.makedirs(out_dir, exist_ok=True)
    
    txt_path = os.path.join(out_dir, 'metrics_report_v2.txt')
    with open(txt_path, 'w') as f:
        f.write("====================================================\n")
        f.write("SmartRoad AI - Final Evaluation Metrics (100 Episodes)\n")
        f.write("====================================================\n\n")
        
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n")
        
        f.write("Classification Report (Precision / Recall / F1):\n")
        f.write(report)
        f.write("\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        
        f.write(f"Total Evaluated Steps: {len(predictions)}\n")
        f.write(f"Average Attention Score: {np.mean(attention_scores):.2f}/100\n")
        f.write(f"Maximum Attention Score: {np.max(attention_scores):.2f}/100\n")
        
    print(f"Created {txt_path}")

    # 4. Generate Markdown Summary 
    md_path = os.path.join(out_dir, 'evaluation_summary_final.md')
    report_dict = classification_report(ground_truths, predictions, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    
    with open(md_path, 'w') as f:
        f.write("# SmartRoad AI — Final Scenario Evaluation Summary\n\n")
        
        f.write("## Overall Performance\n")
        f.write(f"- **Accuracy:** {acc*100:.2f}%\n")
        f.write(f"- **Total Frames Evaluated:** {len(predictions)}\n")
        f.write(f"- **Avg Attention Score:** {np.mean(attention_scores):.2f}\n")
        f.write(f"- **Max Attention Score:** {np.max(attention_scores):.2f}\n\n")
        
        f.write("## Per-Class Breakdown\n\n")
        f.write("| Action Label | Precision | Recall | F1-Score | Support (Frames) |\n")
        f.write("|--------------|-----------|--------|----------|------------------|\n")
        
        for label in target_names:
            metrics = report_dict.get(label, {})
            f.write(f"| {label} | {metrics.get('precision', 0):.2f} | {metrics.get('recall', 0):.2f} | {metrics.get('f1-score', 0):.2f} | {metrics.get('support', 0)} |\n")
            
        f.write("\n## Confusion Matrix\n")
        f.write("*(Rows = Expected Actual, Columns = Agent Prediction)*\n\n")
        f.write("| | Predicted Clear | Predicted Monitor | Predicted Violation |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Actual Clear** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |\n")
        f.write(f"| **Actual Monitor** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |\n")
        f.write(f"| **Actual Violation** | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |\n\n")
        
        f.write("---\n*Note: Evaluation generated dynamically by mapping observation vectors to ground truth state rules, covering 100 PPO episodes.*")
        
    print(f"Created {md_path}")

if __name__ == '__main__':
    generate_metrics()
