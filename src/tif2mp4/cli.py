"""``tif2mp4 <input.tif> <output.mp4> [--camera beh.mp4]``.

There is one subject: a ScanImage TIFF. Every option describes how to render it.
A behavioural camera video can ride along as a supplement, shown beside it, and
the only thing that happens to that video is that it is put on the TIFF's clock:
scaled to the TIFF panel's height at its own aspect ratio, and down-sampled to
the TIFF's frame rate. No rotation, no smoothing, no overlays -- the stamp and
the marker are already on the TIFF panel, so repeating them would say nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from . import overlay, render
from .sources import STIMULUS_TYPES, InputError, load_tiff, load_video

FOURCC = "mp4v"


def _ratio(text: str) -> float:
    """Parse an ``N:M`` ratio into a positive fraction."""
    try:
        numerator, denominator = (float(part) for part in text.split(":"))
        value = numerator / denominator
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(f"{text!r} is not an N:M ratio, e.g. 1:3") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tif2mp4",
        description=(
            "Render a ScanImage TIFF to an .mp4, optionally beside the behavioural "
            "camera video from the same trial. All options describe the TIFF."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Native playback (--speed 1):\n"
            "  recording  SI.hRoiManager.scanFrameRate, stamped in seconds\n"
            "  z-stack    one z-step per second, stamped in microns\n"
        ),
    )
    parser.add_argument("input", type=Path, help="the ScanImage .tif/.tiff to render")
    parser.add_argument("output", type=Path, help="the .mp4 to write")
    parser.add_argument(
        "--camera",
        type=Path,
        default=None,
        metavar="MP4",
        help="behavioural camera video to show beside the TIFF, on the TIFF's clock",
    )
    parser.add_argument(
        "--camera-ratio",
        type=_ratio,
        default="1:3",
        metavar="N:M",
        help=(
            "camera height as a fraction of the TIFF's, e.g. 1:3 (the default) "
            "for a third as tall. 1:1 matches the TIFF"
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="play N times faster than native acquisition (default: 1.0, real time)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        metavar="DEG",
        help="rotate the TIFF clockwise by 0, 90, 180 or 270 degrees (default: 0)",
    )
    parser.add_argument(
        "--running-average",
        type=int,
        default=render.RUNNING_AVERAGE,
        metavar="N",
        help=(
            f"running-average window in TIFF frames (default: {render.RUNNING_AVERAGE}, "
            f"no smoothing). Try 3 for a recording; leave it alone for a z-stack, "
            f"whose frames are depths rather than time points"
        ),
    )
    parser.add_argument(
        "--stimulus-type",
        choices=STIMULUS_TYPES,
        default=None,
        help="stimulus to mark on the TIFF panel; no marker unless given",
    )
    parser.add_argument(
        "--stimulus-onset",
        type=float,
        default=None,
        metavar="SEC",
        help="stimulus onset in real seconds",
    )
    parser.add_argument(
        "--stimulus-duration",
        type=float,
        default=None,
        metavar="SEC",
        help="stimulus duration in real seconds",
    )
    args = parser.parse_args(argv)
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.running_average < 1:
        parser.error("--running-average must be at least 1")

    # A stimulus is described in full or not at all: no timing is assumed.
    stimulus = {
        "--stimulus-type": args.stimulus_type,
        "--stimulus-onset": args.stimulus_onset,
        "--stimulus-duration": args.stimulus_duration,
    }
    missing = [flag for flag, value in stimulus.items() if value is None]
    if missing and len(missing) < len(stimulus):
        parser.error(
            "a stimulus must be described in full: "
            + ", ".join(sorted(missing))
            + " also needed"
        )
    if args.stimulus_duration is not None and args.stimulus_duration <= 0:
        parser.error("--stimulus-duration must be positive")
    return args


def build(args: argparse.Namespace) -> None:
    for path in (args.input, args.camera):
        if path is not None and not path.exists():
            raise InputError(f"{path}: no such file")

    print(f"Reading {args.input.name}:")
    tiff = load_tiff(args.input)
    if tiff.rate is None:
        raise InputError(f"{args.input}: ScanImage header has no usable rate")
    print(
        f"  {tiff.n_frames} frames, {tiff.shape[1]}x{tiff.shape[0]}, "
        f"{tiff.kind}, {tiff.rate:.6g} Hz native"
        + (f", {tiff.step_um:.4g} um/frame" if tiff.step_um else "")
    )

    is_zstack = tiff.kind == "zstack"
    if is_zstack and args.stimulus_type is not None:
        raise InputError("a z-stack has no time axis, so a stimulus cannot be placed on it")

    # The TIFF is the clock: one output frame per TIFF frame.
    n_out = tiff.n_frames
    out_fps = tiff.rate * args.speed

    lo, hi = render.PERCENTILES
    print(f"Contrast-stretching each frame on its own {lo}-{hi}th pct ...")
    stack = render.stretch(tiff.frames)

    if args.running_average > 1:
        print(f"Averaging over {args.running_average} frames ...")
    stack = render.smooth(stack, args.running_average)

    panel = render.to_bgr(stack)
    if args.rotate:
        panel = render.rotate(panel, args.rotate)
        print(f"Rotating {args.rotate} deg ...")
    height = panel.shape[1]

    camera = camera_panel = None
    if args.camera is not None:
        if is_zstack:
            raise InputError("a z-stack has no time axis to put a camera video on")
        print(f"Reading {args.camera.name}:")
        camera = load_video(args.camera)
        print(f"  {camera.n_frames} frames, {camera.shape[1]}x{camera.shape[0]}")

        # The camera's pixels are left alone: it is already display-ready, so
        # only its geometry and its clock change.
        camera_height = max(1, int(round(height * args.camera_ratio)))
        scaled = [render.to_height(frame, camera_height) for frame in camera.frames]
        print(f"Scaling the camera to {args.camera_ratio:.3g} of TIFF height ...")

        # Pad into a full-height column so the two panels concatenate.
        camera_panel = np.stack([render.in_column(frame, height) for frame in scaled])
        print("Down-sampling the camera onto the TIFF's clock:")
        print(f"  {camera.n_frames} -> {n_out} frames")

    if args.speed != 1:
        # Only the encode rate changes: no frame is added, dropped or resampled.
        # A stack is traversed, not played: its rate is depth per second.
        unit = "z-steps/s" if is_zstack else "fps"
        print(f"{'Speeding up' if args.speed > 1 else 'Slowing down'} {args.speed:.4g}x:")
        print(f"  {tiff.rate:.6g} -> {out_fps:.4g} {unit}")

    summary = (
        f"{tiff.step_um * args.speed:.4g} um/s" if is_zstack else f"{args.speed:.4g}x real time"
    )
    width = panel.shape[2] + (0 if camera_panel is None else camera_panel.shape[2])
    print(f"Summary: encoded {n_out} frames @ {out_fps:.4g} fps ({summary})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*FOURCC), out_fps, (width, height), isColor=True
    )
    if not writer.isOpened():
        raise InputError(f"{args.output}: cannot open for writing")

    try:
        for j in range(n_out):
            t = j / tiff.rate  # real time, independent of --speed
            frame = np.ascontiguousarray(panel[j])

            # Overlays belong to the TIFF, so they are drawn before the camera
            # is joined on and are sized by the TIFF panel alone.
            if is_zstack:
                overlay.draw_stamp(frame, f"{j * tiff.step_um:.0f} um")
            else:
                overlay.draw_stamp(frame, f"{t:.2f} sec")
                if args.stimulus_type is not None:
                    if args.stimulus_onset <= t < args.stimulus_onset + args.stimulus_duration:
                        overlay.draw_stimulus(frame, args.stimulus_type)

            if camera is not None:
                # Down-sample to the TIFF's frame rate: output frame j takes the
                # proportionally-placed camera frame.
                index = min(camera_panel.shape[0] - 1, j * camera_panel.shape[0] // n_out)
                frame = cv2.hconcat([frame, camera_panel[index]])

            writer.write(frame)
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build(args)
    except InputError as error:
        print(f"tif2mp4: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
