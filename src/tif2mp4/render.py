"""Turning raw frames into displayable 8-bit BGR.

TIFF and MP4 get different treatment on purpose.  A ScanImage TIFF is raw 16-bit
sensor data that needs contrast stretching to be legible in a talk, and, if it is
a recording, usually a little temporal smoothing on top.  An MP4 has already been
through a display pipeline, so its
pixels are passed through untouched and only its geometry is changed: nothing
here is worth 28 GB of float64 and a minute and a half of percentiles to redo
what the camera's own encoder already did.
"""

from __future__ import annotations

import cv2
import numpy as np

#: No smoothing unless asked for. A recording usually wants 3; a z-stack wants
#: none at all, since its frames are depths and averaging them blurs z-planes
#: into each other.
RUNNING_AVERAGE = 1
PERCENTILES = (1, 99)


def _stretch_frame(frame: np.ndarray) -> np.ndarray:
    """Percentile-clip to [0, 1]."""
    lo, hi = np.percentile(frame, PERCENTILES[0]), np.percentile(frame, PERCENTILES[1])
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def stretch(frames: np.ndarray) -> np.ndarray:
    """Percentile-clip every frame onto [0, 255], each on its own percentiles.

    Returns float64: the caller may still want to average before quantising.
    """
    stack = np.array(frames, dtype=np.float64)
    for i in range(len(stack)):
        stack[i] = _stretch_frame(stack[i]) * 255.0
    return stack


def smooth(stack: np.ndarray, window: int) -> np.ndarray:
    """Centred running average over ``window`` frames; ``window`` 1 is a no-op."""
    if window <= 1:
        return stack
    smoothed = np.zeros_like(stack)
    half = window // 2
    n = len(stack)
    for i in range(n):
        smoothed[i] = np.mean(stack[max(0, i - half) : min(n, i + half + 1)], axis=0)
    return smoothed


def to_bgr(stack: np.ndarray) -> np.ndarray:
    """Quantise a stretched stack to 8-bit and give every frame three channels."""
    stack = np.clip(stack, 0, 255).astype(np.uint8)
    if stack.ndim == 4:  # already colour, e.g. a non-ScanImage RGB TIFF
        return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in stack])
    return np.stack([cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in stack])




def rotate(frames: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a whole stack clockwise by 0, 90, 180 or 270 degrees.

    A rotation, not a flip: it changes the viewing angle without mirroring, so
    left-right relationships in the data survive. 90 and 270 transpose the frame,
    so callers must measure panel geometry after this, not before.
    """
    turns = (degrees // 90) % 4
    if turns == 0:
        return frames
    # np.rot90 turns counter-clockwise, so negate for a clockwise rotation.
    return np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))


def in_column(frame: np.ndarray, height: int) -> np.ndarray:
    """Sit a short frame at the top of a black column of ``height``.

    The column is what makes the two panels concatenable; flushing the frame to
    its top is what puts the camera beside the TIFF's top-right corner rather
    than floating in the middle of the empty space.
    """
    h = frame.shape[0]
    if h >= height:
        return frame
    column = np.zeros((height, frame.shape[1], frame.shape[2]), dtype=frame.dtype)
    column[:h] = frame
    return column


def to_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Scale a panel to a common height, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    interpolation = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interpolation)
