"""
eye_tracker.py — wraps MediaPipe FaceLandmarker (Tasks API, v0.10+) to detect
whether the user is focused on their screen.

No Qt imports here. Pure Python + OpenCV + MediaPipe.

How "focused" is determined (3-layer priority):
  1. No face detected      → distracted (user left the frame)
  2. Head yaw > YAW_LIMIT  → distracted (user turned head away)
  3. EAR < threshold       → distracted (eyes closed too long)
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_MODEL_PATH = Path(__file__).parent.parent / "assets" / "face_landmarker.task"


# Landmark index constants (all referenced from /assets)
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

_NOSE_TIP   = 4
_LEFT_EDGE  = 234
_RIGHT_EDGE = 454
_FOREHEAD   = 10  
_CHIN       = 152 

YAW_LIMIT_DEGREES   = 30.0
PITCH_LIMIT_DEGREES = 19.0 
BLINK_FRAMES        = 20


class EyeTracker:
    """
    Stateful eye tracker. Call process_frame() each webcam frame.

    Usage:
        tracker = EyeTracker()
        result  = tracker.process_frame(frame, threshold=0.22)
        if not result["is_focused"]:
            ...
        tracker.close()  # always call this when done
    """

    def __init__(self, model_path: Path = _MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Run: curl -sL https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                f"-o {model_path}"
            )
        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,  # one frame at a time (sync)
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)
        self._closed_frame_count = 0

    def close(self):
        """Release MediaPipe resources. Always call this when done."""
        self._detector.close()



    # Main public method

    def process_frame(self, frame, threshold: float = 0.22) -> dict:
        """
        Analyse one BGR frame (as returned by cv2.VideoCapture.read()).

        Returns a dict:
          is_focused    bool  — True if no distraction detected
          face_detected bool  — False if MediaPipe found no face
          ear           float — averaged Eye Aspect Ratio (0 if no face)
          yaw_degrees   float — estimated head yaw (0 if no face)
          reason        str   — human-readable reason if not focused
        """
        h, w = frame.shape[:2]

        # MediaPipe Tasks API expects an mp.Image in SRGB format.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            self._closed_frame_count = 0
            return {
                "is_focused":    False,
                "face_detected": False,
                "ear":           0.0,
                "yaw_degrees":   0.0,
                "pitch_degrees": 0.0,
                "reason":        "No face detected",
            }

        # result.face_landmarks is a list-of-faces; each face is a list of
        # NormalizedLandmark objects with .x, .y, .z in [0, 1].
        landmarks = result.face_landmarks[0]

        # --- Layer 2a: head yaw (left/right) -------------------------
        yaw = self._compute_yaw(landmarks, w)
        if abs(yaw) > YAW_LIMIT_DEGREES:
            self._closed_frame_count = 0
            return {
                "is_focused":   False,
                "face_detected": True,
                "ear":          0.0,
                "yaw_degrees":  yaw,
                "pitch_degrees": 0.0,
                "reason":       f"Head turned ({yaw:.0f}°)",
            }

        # --- Layer 2b: head pitch (up/down) ---------------------------
        pitch = self._compute_pitch(landmarks, h)
        if abs(pitch) > PITCH_LIMIT_DEGREES:
            self._closed_frame_count = 0
            return {
                "is_focused":   False,
                "face_detected": True,
                "ear":          0.0,
                "yaw_degrees":  yaw,
                "pitch_degrees": pitch,
                "reason":       f"Head tilted ({pitch:.0f}°)",
            }

        # --- Layer 3: EAR (eye closure) --------------------------------
        ear = self._compute_ear(landmarks, w, h)
        if ear < threshold:
            self._closed_frame_count += 1
        else:
            self._closed_frame_count = 0

        eyes_closed = self._closed_frame_count >= BLINK_FRAMES

        return {
            "is_focused":    not eyes_closed,
            "face_detected": True,
            "ear":           ear,
            "yaw_degrees":   yaw,
            "pitch_degrees": pitch,
            "reason":        "Eyes closed" if eyes_closed else "",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_ear(self, landmarks, w: int, h: int) -> float:
        left  = self._eye_aspect_ratio(landmarks, _LEFT_EYE,  w, h)
        right = self._eye_aspect_ratio(landmarks, _RIGHT_EYE, w, h)
        return (left + right) / 2.0

    @staticmethod
    def _eye_aspect_ratio(landmarks, indices: list[int], w: int, h: int) -> float:
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        p1 = left corner  p4 = right corner
        p2/p3 = upper lid  p5/p6 = lower lid
        """
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32,
        )
        A = np.linalg.norm(pts[1] - pts[5])   # vertical 1
        B = np.linalg.norm(pts[2] - pts[4])   # vertical 2
        C = np.linalg.norm(pts[0] - pts[3])   # horizontal
        return (A + B) / (2.0 * C + 1e-6)

    @staticmethod
    def _compute_yaw(landmarks, w: int) -> float:
        """
        Approximate head yaw in degrees from nose tip offset relative to
        the midpoint between the left and right face edges.

        Positive = turned right, negative = turned left.
        """
        nose_x  = landmarks[_NOSE_TIP].x  * w
        left_x  = landmarks[_LEFT_EDGE].x * w
        right_x = landmarks[_RIGHT_EDGE].x * w

        face_width  = right_x - left_x
        face_center = (left_x + right_x) / 2.0

        if face_width < 1:
            return 0.0

        offset_ratio = (nose_x - face_center) / (face_width / 2.0)
        return offset_ratio * 45.0

    @staticmethod
    def _compute_pitch(landmarks, h: int) -> float:
        """
        Approximate head pitch in degrees — same idea as yaw but vertical.

        We measure how far the nose has drifted from the midpoint between
        forehead (landmark 10) and chin (landmark 152).

        Image coordinates: y=0 is the TOP, increases downward.
          Positive pitch = nose below centre = looking DOWN (phone)
          Negative pitch = nose above centre = looking UP
        """
        nose_y     = landmarks[_NOSE_TIP].y  * h
        forehead_y = landmarks[_FOREHEAD].y  * h
        chin_y     = landmarks[_CHIN].y      * h

        face_height   = chin_y - forehead_y
        face_v_center = (forehead_y + chin_y) / 2.0

        if face_height < 1:
            return 0.0

        offset_ratio = (nose_y - face_v_center) / (face_height / 2.0)
        return offset_ratio * 45.0
