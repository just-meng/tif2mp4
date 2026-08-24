# tif2mp4

Render a ScanImage TIFF to a presentation-ready `.mp4`, optionally beside the
behavioural camera video from the same trial.

## Installation

```sh
uv tool install git@github.com:just-meng/tif2mp4.git
```

Or from a clone, with `--editable` so edits take effect on the next run. Re-run
it after moving the clone, or after changing dependencies or entry points:

```sh
uv tool install --editable .
```

## Usage

```
tif2mp4 <input.tif> <output.mp4> [--camera beh.mp4]
```

```sh
datalad run -m "render trial 13" \
  -i rec.tif -i cam.mp4 -o out.mp4 \
  "tif2mp4 {inputs[0]} {outputs} --camera {inputs[1]} --rotate 180"
```

### `--speed N`

Plays the result N times faster than it was acquired. `--speed 1` (the default)
means "as it happened". Native rate is read from the header:

| TIFF | native rate (`--speed 1`) | frames stamped with |
|---|---|---|
| recording | `SI.hRoiManager.scanFrameRate` | seconds |
| z-stack | one z-step per second | microns |

For a z-stack, `--speed N` therefore means `N × stackZStepSize` µm/s.

### `--rotate DEG`

Rotates clockwise by `0`, `90`, `180` or `270` degrees; default `0`.

### `--running-average N`

Running-average window in TIFF frames, default `1` — no smoothing unless asked
for. `3` is a good starting point for a recording.

Leave it at the default for a z-stack. Its frames are depths, not time points,
so a running average would blend adjacent z-planes into each other; each slice
is also already an average of the frames acquired at that depth.

### Stimulus marker

```sh
tif2mp4 rec.tif out.mp4 --stimulus-type airpuff --stimulus-onset 2 --stimulus-duration 1
```

`--stimulus-type {airpuff,sound}`, `--stimulus-onset SEC` and
`--stimulus-duration SEC` mark the stimulus on the TIFF panel. 

### `--camera MP4`

Shows a behavioural camera video beside the TIFF, down-sampled to the TIFF's frame rate.

`--camera-ratio N:M` sets its height as a fraction of the TIFF's, default `1:3`.

## What it does to the pixels

A ScanImage TIFF is raw 16-bit sensor data, so it gets a 1–99th percentile
contrast stretch per frame, then `--rotate`. A running average over
`--running-average` frames is applied in between, if you ask for one.

A camera video is already display-ready, so its pixels are copied through
unchanged and only its size and frame rate are touched.

