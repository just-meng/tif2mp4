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

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4"}
TABLE_SUFFIXES = {".tsv"}

#: Fallback for a trials.tsv with no .json sidecar, per the raw/ dataset.
DEFAULT_TRIAL_LEVELS = {"1": "none", "2": "airpuff", "3": "sound"}

STIMULUS_TYPES = ("none", "airpuff", "sound")


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
    container_fps: float | None = None  # what an .mp4 claims, reliable or not
    notes: list[str] = field(default_factory=list)

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
    with tifffile.TiffFile(path) as tf:
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
    container_fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise InputError(f"{path}: contains no decodable frames")
    # rate stays None: acquisition rate is not recoverable from the container.
    return Source(path, np.stack(frames), "video", container_fps=container_fps)


def load_source(path: Path) -> Source:
    suffix = path.suffix.lower()
    if suffix in TIFF_SUFFIXES:
        return load_tiff(path)
    if suffix in VIDEO_SUFFIXES:
        return load_video(path)
    raise InputError(f"{path}: unsupported input type {suffix!r}")


# --------------------------------------------------------------------------- #
# trials.tsv
# --------------------------------------------------------------------------- #


def trial_id_from_name(path: Path) -> int | None:
    """Pull the trailing trial number out of e.g. ``..._trial-airpuff_00013.tif``."""
    match = re.search(r"_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _find_sidecar(table: Path) -> Path | None:
    """Locate the .json describing a table, searching upward per BIDS inheritance."""
    name = table.with_suffix(".json").name
    for directory in [table.parent, *table.parent.parents]:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def stimulus_from_trials(table: Path, trial_id: int) -> str:
    """Look up the stimulus label for ``trial_id``, per trials.tsv and its sidecar."""
    levels = dict(DEFAULT_TRIAL_LEVELS)
    sidecar = _find_sidecar(table.resolve())
    if sidecar is not None:
        described = json.loads(sidecar.read_text()).get("trial_type", {}).get("Levels")
        if described:
            levels = {str(k): str(v) for k, v in described.items()}

    with table.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("trial_id", "").strip() != str(trial_id):
                continue
            code = row.get("trial_type", "").strip()
            label = levels.get(code)
            if label is None:
                raise InputError(f"{table}: trial {trial_id} has unknown trial_type {code!r}")
            label = label.strip().lower().replace(" ", "")
            if label in ("nostim", "none", ""):
                return "none"
            if label not in STIMULUS_TYPES:
                raise InputError(f"{table}: trial {trial_id} stimulus {label!r} is not drawable")
            return label
    raise InputError(f"{table}: no row for trial_id {trial_id}")
