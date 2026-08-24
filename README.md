# tif2mp4

Render ScanImage TIFF stacks and companion camera videos to presentation-ready
`.mp4` files. Several inputs are composited side by side on a shared real-time
axis.

Every rendering decision is read from the files themselves — frame rate, z-step,
whether the file is a stack or a recording, what unit to stamp frames with. The
only knob is how fast the result plays.

## Installation

```sh
uv tool install /mnt/Data/et_psychedelics/dissemination/09_Supervision_Committee_Meeting_2026-03-11/tif2mp4
```

This puts `tif2mp4` on `PATH`, which is what lets `datalad run` invoke it. To
upgrade after editing the source, add `--force`. To run without installing:

```sh
uv run tif2mp4 <input>... <output>      # from inside this directory
```

## Usage

```
tif2mp4 <input>... <output>
```

Inputs are variadic and the **last positional is the output**. That shape is
deliberate: under `datalad run`, `{inputs}` expands to every declared input,
space-separated, and `{outputs}` to the file being written, so the command needs
no rewriting between a one-panel and a two-panel render.

Accepted inputs are `.tif`/`.tiff` (ScanImage), `.mp4`, and optionally a
`trials.tsv`.

### `--speed` — the only rendering option

`--speed N` plays the result N times faster than it was acquired. `--speed 1`
(the default) always means "as it happened". What *native* means is read from
the metadata:

| input | native rate (`--speed 1`) | frames stamped with |
|---|---|---|
| recording TIFF | `SI.hRoiManager.scanFrameRate` | seconds |
| z-stack TIFF | one z-step per second | microns |
| `.mp4` alongside a TIFF | frame count ÷ the TIFF's duration | seconds |
| `.mp4` alone | unknown — see below | nothing |

For a z-stack, `--speed N` therefore means `N × stackZStepSize` µm/s.

### The other options

`--acq-rate HZ` supplies the acquisition rate for an `.mp4`. It is needed
because **MP4 containers do not record it**: the camera file in `inputs/`
declares a uniform 30 fps (`stts`: 914 samples, every delta 1000) that is simply
the Windows Media Foundation writer's default. The real rate is recoverable only
by pairing the video with the TIFF from the same trial. Passed alone with no
rate, an `.mp4` renders unstamped, since there is no honest time to print.

> For this rig's facial camera the true rate is **91.3 Hz** (914 frames over the
> TIFF's 10.011 s).

`--stimulus-type {none,airpuff,sound}`, `--stimulus-onset SEC` (default 2.0) and
`--stimulus-duration SEC` (default 1.0) place the stimulus marker. If a
`trials.tsv` is among the inputs it supplies the *type* authoritatively — looked
up by the trial number in the input filename (`..._00013.tif` → trial 13) and
decoded via the `trial_type` levels in the nearest `trials.json`. Onset and
duration stay options, because the trials table records neither. Passing both a
table and `--stimulus-type` is an error rather than a silent precedence rule.

## Regenerating the committed outputs

```sh
datalad run -m "render 2p recording for presentation" \
  --input inputs/sub-240226O_exp-LSD_ses-pre_trial-airpuff_00013.tif \
  --output outputs/example_recording_2p.mp4 \
  'tif2mp4 {inputs} {outputs} --stimulus-type airpuff'
```

```sh
datalad run -m "render z-stack for presentation" \
  --input inputs/sub-240226O_z-stack.tif \
  --output outputs/example_z-stack.mp4 \
  'tif2mp4 {inputs} {outputs} --speed 5'
```

```sh
datalad run -m "render camera recording for presentation" \
  --input inputs/sub-240226O_exp-LSD_ses-pre_trial-airpuff_00013.mp4 \
  --output outputs/example_recording_camera.mp4 \
  'tif2mp4 {inputs} {outputs} --acq-rate 91.3 --stimulus-type airpuff'
```

Side by side, the case the variadic shape exists for — the camera's rate is
derived from the TIFF, so no `--acq-rate` is needed:

```sh
datalad run -m "render side-by-side 2p + camera for presentation" \
  --input inputs/sub-240226O_exp-LSD_ses-pre_trial-airpuff_00013.tif \
  --input inputs/sub-240226O_exp-LSD_ses-pre_trial-airpuff_00013.mp4 \
  --output outputs/example_side-by-side.mp4 \
  'tif2mp4 {inputs} {outputs} --stimulus-type airpuff'
```

## What it does to the pixels

TIFF and MP4 are treated differently on purpose. A ScanImage TIFF is raw 16-bit
sensor data, so it gets a 1–99th percentile contrast stretch per frame, a
vertical flip, and a 3-frame running average. An `.mp4` has already been through
a display pipeline, so it only gets the contrast stretch.

Panels in a composite are scaled to the **smallest** panel height, so no source
is ever upsampled, and the fastest source sets the output grid so none is
temporally undersampled. Overlay sizes derive from panel height (tuned at
512 px), rather than each render carrying its own constants.

## Notes on the ScanImage header

`numSlices` and `stackZStepSize` are GUI settings that persist whether or not a
stack was acquired — both TIFFs in `inputs/` report `numSlices = 21`, and the
plain recording reports `stackZStepSize = 2` despite being a single plane. The
fields describing what actually happened, and the ones this tool reads, are
`enable`, `actualNumSlices` and `actualStackZStepSize`.
