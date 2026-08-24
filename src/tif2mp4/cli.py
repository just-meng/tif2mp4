"""``tif2mp4 <input>... <output>`` -- render recordings and stacks for talks.

Inputs are variadic and the last positional is the output, which is what lets
the command sit under ``datalad run`` unchanged: ``{inputs}`` expands to every
declared input, space-separated, and ``{outputs}`` to the file being produced.

Every rendering decision is read from the files themselves.  The single knob is
``--speed``: how many times faster than native the result plays.  Native is
whatever the metadata says it is -- the acquisition rate for a recording, one
z-step per second for a stack -- so ``--speed 1`` always means "as it happened".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from . import overlay, render
from .sources import (
    STIMULUS_TYPES,
    TABLE_SUFFIXES,
    InputError,
    Source,
    load_source,
    stimulus_from_trials,
    trial_id_from_name,
)

FOURCC = "mp4v"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tif2mp4",
        description=(
            "Render ScanImage TIFF stacks and companion camera videos to an .mp4. "
            "Several inputs are composited side by side on a shared real-time axis. "
            "A trials.tsv among the inputs supplies the stimulus type."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Native playback (--speed 1) per input type:\n"
            "  recording TIFF  SI.hRoiManager.scanFrameRate, stamped in seconds\n"
            "  z-stack TIFF    one z-step per second, stamped in microns\n"
            "  .mp4 + TIFF     frame count divided by the TIFF's duration\n"
            "  .mp4 alone      unknown -- unstamped unless --acq-rate is given\n"
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="input... output",
        help="one or more inputs (.tif/.tiff/.mp4, plus an optional trials.tsv), then the output .mp4",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="play N times faster than native acquisition (default: 1.0, real time)",
    )
    parser.add_argument(
        "--acq-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="acquisition rate for .mp4 inputs, whose containers do not record it",
    )
    parser.add_argument(
        "--stimulus-type",
        choices=STIMULUS_TYPES,
        default=None,
        help="stimulus to mark when no trials.tsv is given (default: none)",
    )
    parser.add_argument(
        "--stimulus-onset",
        type=float,
        default=2.0,
        metavar="SEC",
        help="stimulus onset in real seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stimulus-duration",
        type=float,
        default=1.0,
        metavar="SEC",
        help="stimulus duration in real seconds (default: 1.0)",
    )
    args = parser.parse_args(argv)
    if len(args.paths) < 2:
        parser.error("need at least one input and an output path")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.acq_rate is not None and args.acq_rate <= 0:
        parser.error("--acq-rate must be positive")
    return args


def resolve_clocks(sources: list[Source], acq_rate: float | None) -> None:
    """Give every video source a rate, from --acq-rate or from a TIFF's duration."""
    reference = next(
        (s for s in sources if s.kind == "recording" and s.duration is not None), None
    )
    for source in sources:
        if source.kind != "video":
            continue
        if acq_rate is not None:
            source.rate = acq_rate
            source.notes.append(f"rate {acq_rate:.4g} Hz from --acq-rate")
        elif reference is not None:
            source.rate = source.n_frames / reference.duration
            source.notes.append(
                f"rate {source.rate:.4g} Hz derived from {reference.path.name} "
                f"({source.n_frames} frames over {reference.duration:.4g} s)"
            )
        else:
            source.notes.append(
                "no clock: .mp4 containers do not record the acquisition rate, and no "
                "TIFF or --acq-rate was given -- rendering without stamps"
            )


def check_shared_axis(sources: list[Source]) -> None:
    """Composites must share one real-time axis; say so loudly when they do not."""
    if len(sources) < 2:
        return
    unclocked = [s for s in sources if s.rate is None]
    if unclocked:
        names = ", ".join(s.path.name for s in unclocked)
        raise InputError(
            f"cannot composite {names}: no time axis. Pass the TIFF from the same "
            f"trial, or state the rate with --acq-rate"
        )
    if any(s.kind == "zstack" for s in sources):
        raise InputError(
            "a z-stack has a depth axis, not a time axis, and cannot be composited "
            "with recordings"
        )
    slowest = min(s.rate for s in sources)
    spans = [s.duration for s in sources]
    if max(spans) - min(spans) > 1.0 / slowest:
        detail = "\n".join(
            f"  {s.path.name}: {s.n_frames} frames, {s.rate:.4g} Hz, {s.duration:.4g} s"
            for s in sources
        )
        raise InputError("inputs do not span the same real-time window:\n" + detail)


def resolve_stimulus(args: argparse.Namespace, tables: list[Path], sources: list[Source]) -> str:
    if tables and args.stimulus_type is not None:
        raise InputError(
            "--stimulus-type conflicts with the trials table; drop one so the "
            "stimulus has a single source of truth"
        )
    if tables:
        if len(tables) > 1:
            raise InputError("more than one trials table given")
        named = next((s for s in sources if trial_id_from_name(s.path) is not None), None)
        if named is None:
            raise InputError(
                f"{tables[0]}: cannot tell which trial to look up -- no input filename "
                f"ends in a trial number (e.g. ..._00013.tif)"
            )
        trial_id = trial_id_from_name(named.path)
        stimulus = stimulus_from_trials(tables[0], trial_id)
        print(f"  stimulus: {stimulus} (trial {trial_id} of {tables[0].name})")
        return stimulus
    return args.stimulus_type or "none"


def build(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.paths]
    output, inputs = paths[-1], paths[:-1]
    tables = [p for p in inputs if p.suffix.lower() in TABLE_SUFFIXES]
    visual = [p for p in inputs if p not in tables]
    if not visual:
        raise InputError("no .tif/.tiff/.mp4 input given")
    for path in inputs:
        if not path.exists():
            raise InputError(f"{path}: no such file")

    sources = [load_source(path) for path in visual]
    resolve_clocks(sources, args.acq_rate)
    check_shared_axis(sources)

    for source in sources:
        detail = f"{source.n_frames} frames, {source.shape[1]}x{source.shape[0]}, {source.kind}"
        if source.rate is not None:
            detail += f", {source.rate:.6g} Hz native"
        if source.step_um is not None:
            detail += f", {source.step_um:.4g} um/frame"
        print(f"  {source.path.name}: {detail}")
        for note in source.notes:
            print(f"    note: {note}")

    stimulus = resolve_stimulus(args, tables, sources)
    is_zstack = sources[0].kind == "zstack"
    if is_zstack and stimulus != "none":
        raise InputError("a z-stack has no time axis, so a stimulus cannot be placed on it")

    # The fastest source sets the output grid so nothing is temporally undersampled.
    base = max(sources, key=lambda s: s.rate or 0.0)
    clocked = base.rate is not None
    base_rate = base.rate if clocked else (base.container_fps or 30.0)
    if not clocked:
        print(
            f"  warning: no clock; replaying the container's {base_rate:.4g} fps and "
            f"omitting stamps"
        )
    out_fps = base_rate * args.speed
    n_out = base.n_frames

    prepared = [
        render.prepare_tiff(s.frames) if s.kind != "video" else render.prepare_video(s.frames)
        for s in sources
    ]
    height = min(frames.shape[1] for frames in prepared)
    panels = [
        np.stack([render.to_height(frame, height) for frame in frames]) for frames in prepared
    ]

    if is_zstack:
        summary = f"{sources[0].step_um * args.speed:.4g} um/s"
    elif clocked:
        summary = f"{args.speed:.4g}x real time"
    else:
        summary = f"{args.speed:.4g}x container rate"
    print(f"  output: {n_out} frames @ {out_fps:.4g} fps ({summary})")

    width = sum(p.shape[2] for p in panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*FOURCC), out_fps, (width, height), isColor=True
    )
    if not writer.isOpened():
        raise InputError(f"{output}: cannot open for writing")

    try:
        for j in range(n_out):
            t = j / base_rate  # real time, independent of --speed
            tiles = []
            for source, frames in zip(sources, panels):
                index = j if source is base else min(frames.shape[0] - 1, int(t * source.rate))
                tiles.append(frames[index])
            frame = tiles[0] if len(tiles) == 1 else cv2.hconcat(tiles)
            frame = np.ascontiguousarray(frame)

            if is_zstack:
                overlay.draw_stamp(frame, f"{j * sources[0].step_um:.0f} um")
            elif clocked:
                overlay.draw_stamp(frame, f"{t:.2f} sec")

            if clocked and stimulus != "none":
                if args.stimulus_onset <= t < args.stimulus_onset + args.stimulus_duration:
                    overlay.draw_stimulus(frame, stimulus)

            writer.write(frame)
    finally:
        writer.release()
    print(f"  wrote {output}")


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
