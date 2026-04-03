"""
main.py — terminal-style cv2 loop.
Keys: [P] pause/resume  [Q] quit
"""

import random
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

from core.eye_tracker import EyeTracker
from core.session_manager import SessionManager
from utils.time_utils import now_iso

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP_FRAMES = 15
CANVAS_W, CANVAS_H = 800, 500
CAM_W, CAM_H       = 240, 180

# Jagged flash: hard-cut between red and dark every N frames while distracted
FLASH_PERIOD = 6
FLASH_ON     = 3

# ---------------------------------------------------------------------------
# Alarm — decode MP3 once at startup; sd.play() hits raw PCM, zero latency
# ---------------------------------------------------------------------------

_ALARM_PATH = Path(__file__).parent / "assets" / "alarm.mp3"
_BEEP, _SR  = sf.read(str(_ALARM_PATH), dtype="float32")  # loads full file into RAM

_alarm_stream: sd.OutputStream | None = None
_alarm_pos = [0]  # sample position — list so the callback can mutate it

def _alarm_callback(outdata, frames, _time, _status):
    """Audio thread callback: copies frames from _BEEP, wrapping seamlessly."""
    pos   = _alarm_pos[0]
    total = len(_BEEP)
    wrote = 0
    while wrote < frames:
        chunk = min(frames - wrote, total - pos)
        outdata[wrote:wrote + chunk] = _BEEP[pos:pos + chunk]
        pos    = (pos + chunk) % total
        wrote += chunk
    _alarm_pos[0] = pos

def start_alarm():
    global _alarm_stream
    if _alarm_stream is not None and _alarm_stream.active:
        return
    offset_seconds = random.choice([0.0, 2.0, 4.0])
    _alarm_pos[0] = int(offset_seconds * _SR) % len(_BEEP)
    channels = _BEEP.shape[1] if _BEEP.ndim > 1 else 1
    _alarm_stream = sd.OutputStream(
        samplerate=_SR,
        channels=channels,
        dtype="float32",
        callback=_alarm_callback,
    )
    _alarm_stream.start()

def stop_alarm():
    global _alarm_stream
    if _alarm_stream is not None:
        _alarm_stream.stop()
        _alarm_stream.close()
        _alarm_stream = None
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

tracker = EyeTracker()
session = SessionManager()
session.start()

cap = cv2.VideoCapture(0)
for _ in range(WARMUP_FRAMES):
    cap.read()

print("LockAlarm — [P] pause/resume  [Q] quit")

last_tick         = time.monotonic()
distraction_start = None
frame_num         = 0
paused            = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        paused = not paused
        if paused:
            stop_alarm()
        # When unpausing, reset distraction_start so no stale event carries over
        distraction_start = None

    result  = tracker.process_frame(frame)
    focused = result["is_focused"]
    now     = time.monotonic()

    if not paused:
        # Tick session once per second
        if now - last_tick >= 1.0:
            session.tick()
            last_tick = now

        # Distraction state machine
        if not focused:
            if distraction_start is None:
                distraction_start = now
                session.distraction_started(now_iso())
                start_alarm()          # in-memory beep — fires this frame
        else:
            if distraction_start is not None:
                session.distraction_ended(now - distraction_start)
                distraction_start = None
                stop_alarm()
    else:
        # Keep last_tick current so we don't get a burst of ticks on unpause
        last_tick = now

    stats     = session.snapshot()
    frame_num += 1

    # --- Build canvas --------------------------------------------------------
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    if paused:
        canvas[:] = (20, 20, 50)          # dim blue tint while paused
    elif not focused and (frame_num % FLASH_PERIOD) < FLASH_ON:
        canvas[:] = (0, 0, 90)            # hard red — jagged, no fade
    else:
        canvas[:] = (30, 30, 30)

    # Webcam thumbnail bottom-right
    small = cv2.resize(frame, (CAM_W, CAM_H))
    x1, y1 = CANVAS_W - CAM_W - 10, CANVAS_H - CAM_H - 10
    x2, y2 = x1 + CAM_W, y1 + CAM_H
    canvas[y1:y2, x1:x2] = small
    border_col = (0, 220, 0) if focused else (0, 0, 220)
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_col, 2)

    # Status line
    if paused:
        status_text, colour = "PAUSED", (80, 80, 200)
    elif focused:
        status_text, colour = "LOCKED IN", (0, 220, 0)
    else:
        status_text, colour = "NOT LOCKED", (0, 0, 220)
    cv2.putText(canvas, status_text, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, colour, 3)

    # Stats
    cv2.putText(canvas, f"Elapsed:       {fmt(stats['total_seconds'])}",  (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Focus time:    {fmt(stats['focus_seconds'])}",  (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Focus score:   {stats['focus_score']}%",        (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Streak:        {fmt(stats['current_streak'])}",  (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, f"Distractions:  {stats['distraction_count']}",   (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Key hints and debug row
    cv2.putText(canvas, "[P] pause  [Q] quit",
                (30, CANVAS_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1)
    cv2.putText(canvas,
                f"EAR: {result['ear']:.3f}  yaw: {result['yaw_degrees']:.1f}  pitch: {result['pitch_degrees']:.1f}",
                (30, CANVAS_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    cv2.imshow("LockAlarm", canvas)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
stop_alarm()
cap.release()
cv2.destroyAllWindows()
tracker.close()
