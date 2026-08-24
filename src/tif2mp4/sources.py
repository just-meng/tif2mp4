"""Inputs, and the clock each one carries.

A source is a stack of frames plus, where the file's own metadata supports it, a
*native rate*: how fast the frames were really acquired.  Everything the CLI
decides -- z-stack or recording, microns or seconds, what fps to encode at --
follows from that, which is why nothing in this module accepts a user
preference.

The ScanImage header needs care.  ``numSlices`` and ``stackZStepSize`` are GUI
settings that persist whether or not a stack was acquired: both TIFFs in
``inputs/`` report ``numSlices = 21``, and the plain recording reports
``stackZStepSize = 2`` despite being a single plane.  The fields that describe
what actually happened are ``enable``, ``actualNumSlices`` and
``actualStackZStepSize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4"}

#: Stimuli the overlay can draw. Absence of --stimulus-type means no marker.
STIMULUS_TYPES = ("airpuff", "sound")


class InputError(Exception):
    """An input cannot be interpreted; reported to the user without a traceback."""


@dataclass
class Source:
    """One visual input: its frames, and how fast they were acquired."""

    path: Path
    frames: np.ndarray
    kind: str  # "recording" | "zstack" | "video"
    rate: float | None = None  # native frames per real second; None when unknown
    step_um: float | None = None  # microns per frame; z-stacks only

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.frames.shape[1:3]
        return int(h), int(w)

    @property
    def duration(self) -> float | None:
        """Real seconds spanned, or None when the clock is unknown."""
        return None if self.rate is None else self.n_frames / self.rate


# --------------------------------------------------------------------------- #
# ScanImage TIFF
# --------------------------------------------------------------------------- #


def _frame_data(tf: tifffile.TiffFile) -> dict:
    return (tf.scanimage_metadata or {}).get("FrameData", {}) or {}


def _is_zstack(fd: dict) -> bool:
    if fd.get("SI.hStackManager.enable") is not True:
        return False
    slices = fd.get("SI.hStackManager.actualNumSlices")
    return isinstance(slices, (int, float)) and slices > 1


def _step_um(fd: dict, path: Path) -> float:
    step = fd.get("SI.hStackManager.actualStackZStepSize")
    if isinstance(step, (int, float)) and step:
        return float(step)
    zs = fd.get("SI.hStackManager.zs")
    if isinstance(zs, list):
        levels = sorted(set(zs))
        if len(levels) > 1:
            return float(np.median(np.diff(levels)))
    raise InputError(f"{path}: z-stack with no usable step size in its ScanImage header")


def load_tiff(path: Path) -> Source:
    if path.suffix.lower() not in TIFF_SUFFIXES:
        raise InputError(
            f"{path}: the input must be a ScanImage TIFF. A camera video goes in "
            f"--camera, not here"
        )
    try:
        tf = tifffile.TiffFile(path)
    except tifffile.TiffFileError as error:
        raise InputError(f"{path}: not a readable TIFF ({error})") from error
    with tf:
        # Read pages explicitly: ScanImage's stale slice/volume counts make
        # tifffile's own reshape guess unreliable for these files.
        frames = np.stack([page.asarray() for page in tf.pages]).astype(np.float64)
        fd = _frame_data(tf)

    if not fd:
        raise InputError(f"{path}: not a ScanImage TIFF -- no header to read a clock from")

    if _is_zstack(fd):
        step = _step_um(fd, path)
        # Native traversal is one step per second, so --speed N gives N*step um/s.
        return Source(path, frames, "zstack", rate=1.0, step_um=step)

    rate = fd.get("SI.hRoiManager.scanFrameRate")
    if not isinstance(rate, (int, float)) or rate <= 0:
        raise InputError(f"{path}: ScanImage header has no usable scanFrameRate")
    return Source(path, frames, "recording", rate=float(rate))


# --------------------------------------------------------------------------- #
# MP4
# --------------------------------------------------------------------------- #


def load_video(path: Path) -> Source:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(f"{path}: cannot open as video")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise InputError(f"{path}: contains no decodable frames")
    # rate stays None: the TIFF supplies the clock, not the container.
    return Source(path, np.stack(frames), "video")

