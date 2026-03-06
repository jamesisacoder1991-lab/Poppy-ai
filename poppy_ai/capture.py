from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class FrameBundle:
    gray: np.ndarray
    timestamp: float


class ScreenCapture:
    """Captures the primary monitor and downsamples frames for fast training."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[1]

    def grab(self) -> FrameBundle:
        raw = np.array(self._sct.grab(self._monitor))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return FrameBundle(gray=resized, timestamp=time.time())
