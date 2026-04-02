import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from core.eye_tracker import EyeTracker
from core.session_manager import SessionManager
import time


WARMUP_FRAMES = 15

# Main canvas size — this is the window you'll see
CANVAS_W, CANVAS_H = 800, 500

# Small webcam thumbnail in the corner
CAM_W, CAM_H = 240, 180

tracker = EyeTracker()
session = SessionManager()
session.start()
last_tick = time.monotonic()
facecam = cv2.VideoCapture(0)

for i in range(WARMUP_FRAMES):
    facecam.read()

print("Camera open. Press 'q' in the window to quit.")


while True:

    ret, frame = facecam.read()

    if ret == False:
        print("Failed to read from camera.")
        break

    result = tracker.process_frame(frame)

    # Tick the session manager once per second
    if time.monotonic() - last_tick >= 1.0:
        session.tick()
        last_tick = time.monotonic()

    if not result["is_focused"]:
        session.distraction_started("now")
    else:
        session.distraction_ended(0)

    stats = session.snapshot()
    focused = result["is_focused"]

    # Build the canvas — a dark background we draw everything onto
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # dark grey background

    # Paste the small webcam feed into the bottom-right corner
    small = cv2.resize(frame, (CAM_W, CAM_H))  # shrink the frame

    # Calculate where the corner thumbnail should sit (10px padding from edges)
    x1 = CANVAS_W - CAM_W - 10
    y1 = CANVAS_H - CAM_H - 10
    x2 = x1 + CAM_W
    y2 = y1 + CAM_H

    canvas[y1:y2, x1:x2] = small  # paste it in

    # Draw a coloured border around the thumbnail (green=focused, red=distracted)
    border_colour = (0, 220, 0) if focused else (0, 0, 220)
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_colour, 2)

    #draw stats
    colour = (0, 220, 0) if focused else (0, 80, 220)

    status_text = "FOCUSED" if focused else "DISTRACTED"
    cv2.putText(canvas, status_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colour, 3)

    cv2.putText(canvas, f"Focus score:   {stats['focus_score']}%",      (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Focus time:    {stats['focus_seconds']}s",     (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Distractions:  {stats['distraction_count']}",  (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Streak:        {stats['current_streak']}s",    (30, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Small EAR debug info at the bottom (useful while testing)
    cv2.putText(canvas,
                f"EAR: {result['ear']:.3f}  yaw: {result['yaw_degrees']:.1f}°  pitch: {result['pitch_degrees']:.1f}°",
                (30, CANVAS_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    cv2.imshow("LockAlarm", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


facecam.release()
cv2.destroyAllWindows()
tracker.close()
