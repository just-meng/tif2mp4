"""Turning raw frames into displayable 8-bit BGR.

TIFF and MP4 get different treatment on purpose.  A ScanImage TIFF is raw 16-bit
sensor data that needs contrast stretching, the vertical flip that puts the
prism face the right way up, and a little temporal smoothing to be legible in a
talk.  An MP4 has already been through a display pipeline, so it only gets the
contrast stretch.
"""

from __future__ import annotations

import cv2
import numpy as np

AVG_WINDOW = 3
PERCENTILES = (1, 99)


def _stretch(frame: np.ndarray) -> np.ndarray:
    """Percentile-clip to [0, 1]."""
    lo, hi = np.percentile(frame, PERCENTILES[0]), np.percentile(frame, PERCENTILES[1])
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def _running_average(stack: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.zeros_like(stack)
    half = window // 2
    n = len(stack)
    for i in range(n):
        smoothed[i] = np.mean(stack[max(0, i - half) : min(n, i + half + 1)], axis=0)
    return smoothed


def prepare_tiff(frames: np.ndarray, window: int = AVG_WINDOW) -> np.ndarray:
    """Contrast-stretch, flip upside-down, smooth, and return a uint8 BGR stack."""
    stack = np.array(frames, dtype=np.float64)
    for i in range(len(stack)):
        stack[i] = _stretch(stack[i]) * 255.0
    stack = stack[:, ::-1, ...]
    stack = np.clip(_running_average(stack, window), 0, 255).astype(np.uint8)
    return np.stack([cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in stack])


def prepare_video(frames: np.ndarray) -> np.ndarray:
    """Contrast-stretch each already-BGR frame; return a uint8 BGR stack."""
    return np.stack(
        [
            np.clip(_stretch(frame.astype(np.float64)) * 255.0, 0, 255).astype(np.uint8)
            for frame in frames
        ]
    )


def to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Scale a panel to a common height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    interpolation = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interpolation)
