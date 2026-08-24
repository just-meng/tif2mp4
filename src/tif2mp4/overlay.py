"""Frame annotations, sized relative to the frame they are drawn on.

Every constant here was tuned against a 512 px tall frame and is multiplied by
``height / BASE_HEIGHT``, so a 512 px panel reproduces the original hand-tuned
look exactly and larger panels scale with it rather than acquiring their own
set of magic numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

BASE_HEIGHT = 512
FONT = cv2.FONT_HERSHEY_SIMPLEX
STAMP_COLOUR = (255, 255, 255)
STIMULUS_COLOUR = (0, 200, 255)

STAMP_ORIGIN = (10, 20)
STAMP_FONT_SCALE = 0.7
STAMP_THICKNESS = 2

STIMULUS_ORIGIN = (10, 50)
STIMULUS_SIZE = 30
STIMULUS_LABEL_FONT_SCALE = 0.5


def _px(value: float, scale: float) -> int:
    return int(round(value * scale))


def _thickness(value: float, scale: float) -> int:
    return max(1, int(round(value * scale)))


def draw_stamp(frame: np.ndarray, text: str) -> None:
    """Write the time/depth stamp into the top-left corner, in place."""
    scale = frame.shape[0] / BASE_HEIGHT
    cv2.putText(
        frame,
        text,
        (_px(STAMP_ORIGIN[0], scale), _px(STAMP_ORIGIN[1], scale)),
        FONT,
        STAMP_FONT_SCALE * scale,
        STAMP_COLOUR,
        _thickness(STAMP_THICKNESS, scale),
        cv2.LINE_AA,
    )


def draw_stimulus(frame: np.ndarray, kind: str) -> None:
    """Mark an active stimulus below the stamp, in place."""
    scale = frame.shape[0] / BASE_HEIGHT
    x, y = _px(STIMULUS_ORIGIN[0], scale), _px(STIMULUS_ORIGIN[1], scale)
    size = _px(STIMULUS_SIZE, scale)
    if kind == "airpuff":
        _airpuff(frame, x, y, size, scale)
    elif kind == "sound":
        _sound(frame, x, y, size, scale)


def _label(frame: np.ndarray, text: str, x: int, y: int, scale: float) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        STIMULUS_LABEL_FONT_SCALE * scale,
        STIMULUS_COLOUR,
        _thickness(1, scale),
        cv2.LINE_AA,
    )


def _airpuff(frame: np.ndarray, x: int, y: int, size: int, scale: float) -> None:
    """A nozzle with air burst lines, blowing to the right."""
    nozzle = np.array(
        [[x, y], [x + size // 3, y - size // 4], [x + size // 3, y + size // 4]],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [nozzle], STIMULUS_COLOUR)

    burst_x = x + size // 3 + _px(2, scale)
    for dy in (-size // 4, -size // 8, 0, size // 8, size // 4):
        cv2.line(
            frame,
            (burst_x, y + dy),
            (burst_x + size // 2, y + dy + (dy // 2)),
            STIMULUS_COLOUR,
            _thickness(1, scale),
            cv2.LINE_AA,
        )
    _label(frame, "AIR PUFF", x + size + _px(5, scale), y + _px(5, scale), scale)


def _sound(frame: np.ndarray, x: int, y: int, size: int, scale: float) -> None:
    """A speaker with radiating arcs, matching the airpuff icon's footprint."""
    body_w, body_h = size // 4, size // 5
    cone = np.array(
        [
            [x, y - body_h],
            [x, y + body_h],
            [x + body_w, y + size // 4],
            [x + body_w, y - size // 4],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [cone], STIMULUS_COLOUR)

    thickness = _thickness(1, scale)
    for radius in (size // 3, size // 2):
        cv2.ellipse(
            frame,
            (x + body_w, y),
            (radius, radius),
            0,
            -55,
            55,
            STIMULUS_COLOUR,
            thickness,
            cv2.LINE_AA,
        )
    _label(frame, "SOUND", x + size + _px(5, scale), y + _px(5, scale), scale)
