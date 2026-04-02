"""
camera_worker.py — background QThread that drives the webcam, eye tracker,
and session state machine.

Why a QThread and not just a regular loop?
  MediaPipe takes ~30ms per frame. If that ran on the Qt main thread,
  the UI would freeze for 30ms between every frame — buttons feel laggy,
  stats don't update smoothly. Moving it to a QThread keeps the GUI fluid.

How it talks to the UI:
  Only through Qt signals. The UI connects its own functions (slots) to
  these signals. Qt delivers them safely across threads automatically.
  The worker never calls any UI method directly.

Lifecycle:
  worker = CameraWorker()
  worker.some_signal.connect(some_function)  # connect in UI
  worker.start()                             # begins run() on background thread
  ...
  worker.stop()                              # sets flag to exit loop
  worker.wait()                              # blocks until thread actually exits
"""

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from core.eye_tracker import EyeTracker
from core.session_manager import SessionManager
from utils.time_utils import now_iso

# How long the user must look away before the alarm fires (seconds)
DISTRACTION_THRESHOLD = 2.0

# Frames to discard on startup — macOS cameras output black initially
WARMUP_FRAMES = 15

# Calibration: how long to sample the user's open-eye EAR (seconds)
CALIBRATION_DURATION = 5


class CameraWorker(QThread):
    """
    Runs on a background thread. Owns the webcam, EyeTracker, and SessionManager.
    The UI never touches these directly — it only receives signals.
    """

    # ------------------------------------------------------------------
    # Signals — emitted from the background thread, received by the UI
    #
    # pyqtSignal(type) defines what data the signal carries.
    # The UI connects a function to each signal; Qt calls that function
    # on the main thread whenever the signal is emitted here.
    # ------------------------------------------------------------------

    # Calibration phase
    calibration_tick     = pyqtSignal(int)    # countdown: 5, 4, 3, 2, 1
    calibration_complete = pyqtSignal(float)  # the computed EAR threshold
    calibration_failed   = pyqtSignal()       # no face seen during calibration

    # Session phase — ~30fps frame for the webcam preview widget
    frame_ready = pyqtSignal(QImage)

    # Live stats snapshot, emitted every second
    stats_updated = pyqtSignal(dict)

    # Distraction events
    distraction_started = pyqtSignal()
    distraction_ended   = pyqtSignal(float)  # carries duration in seconds
    alarm_triggered     = pyqtSignal()       # fires once after DISTRACTION_THRESHOLD

    # Emitted when the user clicks Stop — carries the full session summary
    session_complete = pyqtSignal(dict)

    # ------------------------------------------------------------------

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.camera_index          = camera_index
        self._running              = False
        self._threshold_multiplier = 1.0   # adjusted by the sensitivity slider

    # ------------------------------------------------------------------
    # Public API (called from the main thread)
    # ------------------------------------------------------------------

    def set_threshold_multiplier(self, multiplier: float):
        """
        Called from the UI sensitivity slider.
        Multiplies the calibrated EAR threshold up or down.
        Float assignment is atomic in CPython so this is thread-safe.
        """
        self._threshold_multiplier = multiplier

    def stop(self):
        """
        Signal the run() loop to exit cleanly.
        Always call worker.wait() after this to block until it's done.
        Never call terminate() — it kills the thread without releasing
        the webcam or MediaPipe resources.
        """
        self._running = False

    # ------------------------------------------------------------------
    # Thread entry point — everything below runs on the background thread
    # ------------------------------------------------------------------

    def run(self):
        """
        Qt calls this when worker.start() is called.
        Do NOT call this directly.
        """
        self._running = True

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return

        # Discard warmup frames — macOS cameras output black initially
        for _ in range(WARMUP_FRAMES):
            cap.read()

        tracker = EyeTracker()

        try:
            # Phase 1: calibration — learn the user's personal EAR threshold
            threshold = self._calibrate(cap, tracker)

            if not self._running:
                return  # user closed the app during calibration

            if threshold is None:
                self.calibration_failed.emit()
                threshold = 0.22  # sensible default

            self.calibration_complete.emit(threshold)

            # Phase 2: active session loop
            self._run_session(cap, tracker, threshold)

        finally:
            # Always runs — even if an exception is thrown above
            tracker.close()
            cap.release()

    # ------------------------------------------------------------------
    # Phase 1: Calibration
    # ------------------------------------------------------------------

    def _calibrate(self, cap, tracker) -> float | None:
        """
        Collect EAR samples for CALIBRATION_DURATION seconds while the
        user looks normally at the screen.

        Returns a personal EAR threshold (mean × 0.75), or None if
        no face was detected during calibration.

        Why 0.75?
          If your open-eye EAR is 0.33, 75% of that = 0.25.
          That sits comfortably between your open-eye baseline and
          the blink level (~0.10), giving a robust threshold.
        """
        ear_samples = []
        start       = time.monotonic()
        last_tick   = -1

        while self._running:
            elapsed = time.monotonic() - start
            if elapsed >= CALIBRATION_DURATION:
                break

            # Emit a countdown number once per second for the UI to display
            seconds_left = CALIBRATION_DURATION - int(elapsed)
            if seconds_left != last_tick:
                self.calibration_tick.emit(seconds_left)
                last_tick = seconds_left

            ret, frame = cap.read()
            if not ret:
                continue

            # threshold=0.0 so the tracker never classifies — just measures
            result = tracker.process_frame(frame, threshold=0.0)
            if result["face_detected"] and result["ear"] > 0.05:
                ear_samples.append(result["ear"])

        if len(ear_samples) < 10:
            return None

        return round(float(np.mean(ear_samples)) * 0.75, 4)

    # ------------------------------------------------------------------
    # Phase 2: Session loop
    # ------------------------------------------------------------------

    def _run_session(self, cap, tracker, calibrated_threshold: float):
        """
        Main loop. Runs until stop() is called.
        Emits signals so the UI can update without any direct coupling.
        """
        session            = SessionManager()
        session.start()

        distraction_start: float | None = None
        alarm_fired                     = False
        last_stats_tick                 = time.monotonic()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            threshold = calibrated_threshold * self._threshold_multiplier
            result    = tracker.process_frame(frame, threshold=threshold)
            focused   = result["is_focused"]
            now       = time.monotonic()

            # --- Distraction / alarm logic ----------------------------
            if not focused:
                if distraction_start is None:
                    distraction_start = now
                    session.distraction_started(now_iso())
                    self.distraction_started.emit()

                if (now - distraction_start) >= DISTRACTION_THRESHOLD and not alarm_fired:
                    self.alarm_triggered.emit()
                    alarm_fired = True
            else:
                if distraction_start is not None:
                    duration = now - distraction_start
                    session.distraction_ended(duration)
                    self.distraction_ended.emit(duration)
                    distraction_start = None
                    alarm_fired       = False

            # --- Send processed frame to the webcam display widget ----
            # We convert to QImage here (on the worker thread) because
            # passing a raw numpy array across a signal boundary is unsafe.
            self.frame_ready.emit(_bgr_to_qimage(frame))

            # --- Stats update once per second -------------------------
            if now - last_stats_tick >= 1.0:
                session.tick()
                self.stats_updated.emit(session.snapshot())
                last_stats_tick = now

        # Loop exited — close any open distraction event
        if distraction_start is not None:
            session.distraction_ended(time.monotonic() - distraction_start)

        # Emit the final session summary to trigger the post-session screen
        self.session_complete.emit(session.finish())


# ---------------------------------------------------------------------------
# Frame conversion utility
# ---------------------------------------------------------------------------

def _bgr_to_qimage(frame) -> QImage:
    """
    Convert an OpenCV BGR numpy array to a QImage safe to pass via signals.

    Steps:
      1. BGR → RGB  (cv2 and Qt use opposite channel order)
      2. Wrap in QImage pointing at the numpy data
      3. .copy() makes an independent copy of the pixel data

    Why .copy()?
      Without it, QImage holds a pointer to the numpy array's memory.
      On the next frame, cap.read() overwrites that memory with new pixels.
      If Qt is still drawing the previous frame at that moment → crash.
      .copy() gives Qt its own independent buffer so this can't happen.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(
        rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
    ).copy()
