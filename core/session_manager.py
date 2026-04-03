"""
session_manager.py — tracks the timing state of a single focus session.

No Qt imports. All timing is managed externally (the camera worker calls
tick() at a fixed interval); this class only counts.

State machine:
  ┌─────────┐  distraction_started()  ┌────────────┐
  │ FOCUSED │ ──────────────────────► │ DISTRACTED │
  │         │ ◄────────────────────── │            │
  └─────────┘  distraction_ended()    └────────────┘
"""

import time
from utils.time_utils import now_iso


class SessionManager:
    """
    Tracks focus time, slack time, streaks, and distraction events
    for one study session.

    Typical usage inside CameraWorker:
        mgr = SessionManager()
        mgr.start()
        ...
        mgr.distraction_started("2026-04-01T09:05:10")
        mgr.distraction_ended(duration_seconds=8.2)
        summary = mgr.finish()
    """

    def __init__(self):
        self.ear_threshold: float = 0.22   # updated after calibration
        self._reset()

    def _reset(self):
        self._started_at: str  = ""
        self._total_ticks: int = 0         # total seconds elapsed
        self._focus_ticks: int = 0         # seconds spent focused
        self._in_distraction: bool = False
        self._current_streak: int  = 0     # focused seconds since last distraction
        self._best_streak: int     = 0
        self._distraction_count: int = 0
        self._distractions: list[dict] = []  # [{occurred_at, duration_seconds}, ...]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Call once when the session begins (after calibration)."""
        self._reset()
        self._started_at = now_iso()

    def finish(self) -> dict:
        """
        Call when the user stops the session. Returns the full summary
        dict that gets passed to db_manager.save_session() and to the
        post-session screen.
        """
        ended_at     = now_iso()
        total        = self._total_ticks
        focus        = self._focus_ticks
        slack        = total - focus
        focus_score  = round((focus / total * 100), 1) if total > 0 else 0.0

        return {
            "started_at":        self._started_at,
            "ended_at":          ended_at,
            "total_seconds":     total,
            "focus_seconds":     focus,
            "slack_seconds":     slack,
            "distraction_count": self._distraction_count,
            "focus_score":       focus_score,
            "best_streak":       self._best_streak,
            "distractions":      list(self._distractions),
        }

    # ------------------------------------------------------------------
    # Per-frame / per-second updates (called from CameraWorker)
    # ------------------------------------------------------------------

    def tick(self):
        """
        Called once per second by the camera worker.
        Increments the right counter depending on current state.
        """
        self._total_ticks += 1
        if not self._in_distraction:
            self._focus_ticks  += 1
            self._current_streak += 1
            self._best_streak = max(self._best_streak, self._current_streak)

    def distraction_started(self, occurred_at: str):
        """
        Call when the eye tracker first detects a distraction.
        Records the event start time for later pairing with distraction_ended().
        """
        if self._in_distraction:
            return  # already in a distraction — ignore re-entry
        self._in_distraction = True
        self._distraction_count += 1
        # Store partial record; duration filled in by distraction_ended().
        self._distractions.append({"occurred_at": occurred_at, "duration_seconds": 0.0})

    def distraction_ended(self, duration_seconds: float):
        """
        Call when the user returns focus. Updates the last distraction record
        with the actual duration and resets the streak counter.
        """
        if not self._in_distraction:
            return
        self._in_distraction = False
        self._current_streak = 0   # streak broken
        if self._distractions:
            self._distractions[-1]["duration_seconds"] = round(duration_seconds, 2)

    # ------------------------------------------------------------------
    # Snapshot (for stats_updated signal — emitted every second)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Returns the current live stats. The camera worker emits this via
        stats_updated signal every second so the UI can update stat cards.
        """
        total = self._total_ticks
        focus = self._focus_ticks
        return {
            "total_seconds":     total,
            "focus_seconds":     focus,
            "slack_seconds":     total - focus,
            "current_streak":    self._current_streak,
            "distraction_count": self._distraction_count,
            "focus_score":       round(focus / total * 100, 1) if total > 0 else 0.0,
            "is_distracted":     self._in_distraction,
        }

