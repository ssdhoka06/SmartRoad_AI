# SmartRoad AI - Test Scenarios

## Scenario 1: Driver using phone while looking ahead
- **Description:** The driver is holding and interacting with a phone device but their face and gaze are directed forward at the road.
- **Expected YOLO Detections:** `phone`, `driver_face`
- **Expected SegFormer Zone:** `driver seat`, `steering wheel area` (partially missing hands), `road view area`
- **Time to Flag:** 3-5 seconds of sustained phone presence.
- **Expected RL Action:** `2` (Violation) - Due to extended active phone usage even if looking ahead.

## Scenario 2: Driver using phone while looking away
- **Description:** The driver is interacting with a phone and looking down or away from the road view towards the phone.
- **Expected YOLO Detections:** `phone` (with `driver_face` profile or missing)
- **Expected SegFormer Zone:** phone in `driver body position` or `dashboard` area.
- **Time to Flag:** 2-3 seconds.
- **Expected RL Action:** `2` (Violation) - High severity distraction (phone + gaze away).

## Scenario 3: Driver sleeping / eyes closed
- **Description:** The driver's head is tilted or eyes are closed for a sustained period without active interaction.
- **Expected YOLO Detections:** `driver_face` (closed eyes/drowsy posture) 
- **Expected SegFormer Zone:** `driver seat`, `driver body position` static.
- **Time to Flag:** 3 seconds.
- **Expected RL Action:** `2` (Violation) - Drowsiness is a critical safety failure.

## Scenario 4: Driver smoking
- **Description:** The driver is holding a cigarette near their face and taking puffs.
- **Expected YOLO Detections:** `cigarette`, `driver_face`
- **Expected SegFormer Zone:** `driver body position`, hand near `driver seat` upper region.
- **Time to Flag:** 5 seconds of sustained smoking activity.
- **Expected RL Action:** `2` (Violation) - Active secondary task reducing vehicle control.

## Scenario 5: Driver fully attentive and safe
- **Description:** Normal driving postured, both hands on or near the steering wheel, gaze directed through the windshield.
- **Expected YOLO Detections:** `driver_face`, `steering wheel`
- **Expected SegFormer Zone:** `driver seat`, `steering wheel area`, `road view area` active.
- **Time to Flag:** N/A
- **Expected RL Action:** `0` (All Clear) - Safe behavior.

## Scenario 6: Driver briefly glancing at phone then away
- **Description:** The driver briefly picks up or glances at a phone (e.g. checking navigation or rejecting a call) for about 1 second, then returns focus to the road.
- **Expected YOLO Detections:** `phone` (briefly), `driver_face`
- **Expected SegFormer Zone:** `driver body position`.
- **Time to Flag:** Initial glance triggers monitor, returns to clear quickly.
- **Expected RL Action:** `1` (Monitor) - Initially flags suspicious activity, but drops back to `0` once normal driving resumes, not triggering a full violation due to short duration.
