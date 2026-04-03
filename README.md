# LockAlarm

Webcam-based focus monitor. Watches your eyes and head position — if you look away, an alarm fires instantly. Terminal-style cv2 window, no GUI bloat.

## What it does
- Detects if you're looking at the screen using MediaPipe Face Landmarks.
- Alarm plays the moment distraction is detected — pre-loaded into RAM, zero latency.
- Tracks focus time, streak, score, and distraction count live
- Flashes red while distracted


## Setup

```bash
pip install -r requirements.txt
```

Download the MediaPipe face landmark model and place it at:
```
assets/face_landmarker.task
```

## Run

```bash
python main.py
```

## Keys

| Key | Action |
|-----|--------|
| `P` | Pause / resume session |
| `Q` | Quit |

---

## How detection works

MediaPipe gives 468 numbered landmark points mapped across the face each frame. LockAlarm uses 11 of them across three detection layers, checked in priority order — if an earlier layer fires, the later ones are skipped for that frame.


### Layer 1 — Face presence
If MediaPipe finds no face in the frame at all → distracted immediately. No further checks needed.


### Layer 2a — Head yaw (left/right turn)
Landmarks: nose tip (`4`), left cheek edge (`234`), right cheek edge (`454`).

Measures how far the nose has shifted from the midpoint between the two cheek edges. When you turn your head, the nose drifts toward one side:

```
offset_ratio = (nose_x - face_center_x) / (face_width / 2)
yaw_degrees  = offset_ratio * 45
```

Fires distraction if `abs(yaw) > 30 degrees`.


### Layer 2b — Head pitch (up/down tilt)
Landmarks: nose tip (`4`), forehead (`10`), chin (`152`).

Same geometry as yaw but vertical. Measures how far the nose has drifted from the midpoint between forehead and chin. Looking down at your phone pushes the nose below center:

```
offset_ratio  = (nose_y - face_v_center) / (face_height / 2)
pitch_degrees = offset_ratio * 45
```

Fires distraction if `abs(pitch) > 19 degrees`.


### Layer 3 — EAR (eye closure)
Landmarks: 6 points per eye — 2 corners, 2 upper lid, 2 lower lid.

```
Left eye: [362, 385, 387, 263, 373, 380]
Right eye: [33, 160, 158, 133, 153, 144]
```

Eye Aspect Ratio formula:

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

`p1`/`p4` are the corners (horizontal span), `p2`/`p3` are upper lid points, `p5`/`p6` are lower lid points. When the eye is open, the vertical distances are large relative to the horizontal width. When the eye closes, those vertical distances collapse and EAR drops toward 0. Both eyes are averaged together.

If EAR stays below `0.22` for **20 consecutive frames** (~650ms at 30fps) → distracted. The 20-frame buffer is what separates a normal blink (~150ms, ~4 frames) from actually closed eyes — blinks pass through without triggering.

---



## Requirements

- Python 3.10+
- Webcam
- macOS (alarm uses `sounddevice` — works cross-platform)
