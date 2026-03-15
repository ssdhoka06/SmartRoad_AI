# SmartRoad AI: System Architecture

## Architecture Diagram

```mermaid
graph TD
    Camera[Webcam / Dashboard Camera] --> Frame[Raw RGB Frame]
    Frame --> SegFormer[Layer 1: SegFormer-b0]
    Frame --> YOLO[Layer 2: YOLOv8 Nano]
    
    SegFormer --> SegMask[Pixel Segmentation Mask]
    SegMask --> ZoneExtraction[Extract Zone Context <br> Driver Seat, Steering, Road]
    
    YOLO --> BBoxes[Object Bounding Boxes]
    BBoxes --> ObjExtraction[Extract Phone, Cigarette, Face, Seatbelt]
    
    ZoneExtraction --> ObsBuilder[`obs_builder.py`]
    ObjExtraction --> ObsBuilder
    
    ObsBuilder --> ObsVector[Observation Vector <br> Shape: 10,]
    
    ObsVector --> PPO[Layer 3: PPO Agent <br> `ppo_v2.zip`]
    PPO --> Action[Decision: All Clear / Monitor / Violation]
    
    Action --> ActionMonitor{Is Violation?}
    ActionMonitor -- Yes --> Logger[`alert_logger.py`]
    Logger --> CSV[Save to violations.csv]
    Logger --> SaveFrame[Save Evidence Frame]
    ActionMonitor -- No --> Continue[Next Frame]
    
    CSV --> EChallan[Challan Generation Pipeline <br> Future Scope]
```

## Module Descriptions

1. **`pipeline.py` & Layer 1/2 Loaders:**
   - Handles webcam initiation and passes frames to HuggingFace Transformers (SegFormer) and Ultralytics (YOLOv8) models.
   - Outputs bounding boxes and semantic masks.

2. **`obs_builder.py`:**
   - Takes combined bounding boxes and segmentation masks to build the reinforcement learning environment's observation vector.
   - Computes state durations (e.g., how long the phone has been actively tracked).

3. **`rl_environment.py`:**
   - The Gym-compatible custom environment (`DriverEnv`) wrapping the state space and the reward logic. Provides the boundary between ML computer vision and RL decision making.

4. **Layer 3 Agent (PPO - `train_ppo.py` / `ppo_v2.zip`):**
   - Given a state vector from the environment, output an action evaluating whether a violation has occurred (`0`, `1`, `2`).

5. **`alert_logger.py`:**
   - If action `2` (Violation) is outputted by the RL agent for a continuous timeframe, it captures the current original RGB frame and logs to a localized CSV with severity tracking.

6. **`final_integrate.py`:**
   - The end-to-end event execution loop. Wraps all the layers sequentially per frame and streams to the visual UI dashboard showing all real-time statuses.
