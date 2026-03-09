#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "Pillow",
#     "opencv-python",
# ]
# ///

import numpy as np
from PIL import Image
import cv2
import argparse

# ============== ARGUMENT PARSING ==============
parser = argparse.ArgumentParser(description="Convert a TIFF stack to .mp4 with smoothing, overlays, and airpuff indicator.")
parser.add_argument("tiff_path", help="Path to the input TIFF stack")
parser.add_argument("output_path", help="Path for the output .mp4 file")
parser.add_argument("--duration", type=float, default=10.0, help="Total video duration in seconds (default: 10.0)")
parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30.0; overrides --duration)")
parser.add_argument("--avg-window", type=int, default=3, help="Running average window size (default: 3)")
parser.add_argument("--puff-start", type=float, default=2.0, help="Airpuff onset time in seconds (default: 2.0)")
parser.add_argument("--puff-duration", type=float, default=1.0, help="Airpuff display duration in seconds (default: 1.0)")
parser.add_argument("--no-flip", action="store_true", help="Disable vertical flip (default: video is flipped upside-down)")
parser.add_argument("--no-enhance", action="store_true", help="Disable percentile contrast enhancement")
parser.add_argument("--no-puff", action="store_true", help="Disable airpuff icon overlay")
parser.add_argument("--overlay", choices=["timestamp", "distance", "none"], default="timestamp",
                    help="Overlay type: 'timestamp' (elapsed time), 'distance' (depth in µm), or 'none' (default: timestamp)")
parser.add_argument("--um-per-frame", type=float, default=5.0, help="Microns per frame for distance overlay (default: 5.0)")
args = parser.parse_args()

tiff_path = args.tiff_path
output_path = args.output_path
total_duration = args.duration
running_avg_window = args.avg_window
airpuff_start = args.puff_start
airpuff_duration = args.puff_duration
flip_vertical = not args.no_flip
enhance = not args.no_enhance

# ============== OVERLAY SETTINGS ==============
font_scale = 0.7
font_color = (255, 255, 255)
timestamp_pos = (10, 20)
airpuff_icon_pos = (10, 50)

# ============== HELPER FUNCTIONS ==============
def enhance_mimg(mimg):
    """Percentile-clip and normalise a mean image to [0, 1]."""
    lo, hi = np.percentile(mimg, 1), np.percentile(mimg, 99)
    return np.clip((mimg - lo) / (hi - lo), 0, 1)

def draw_airpuff_icon(img_bgr, x, y, size=30):
    """Draws a nozzle with air burst lines and 'AIR PUFF' label."""
    nozzle_pts = np.array([
        [x, y],
        [x + size // 3, y - size // 4],
        [x + size // 3, y + size // 4]
    ], dtype=np.int32)
    cv2.fillPoly(img_bgr, [nozzle_pts], (0, 200, 255))

    line_start_x = x + size // 3 + 2
    for dy in [-size // 4, -size // 8, 0, size // 8, size // 4]:
        start_pt = (line_start_x, y + dy)
        end_pt = (line_start_x + size // 2, y + dy + (dy // 2))
        cv2.line(img_bgr, start_pt, end_pt, (0, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(img_bgr, "AIR PUFF", (x + size + 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return img_bgr

# --- 1. Load TIFF stack ---
print(f"Loading TIFF stack from: {tiff_path}")
img = Image.open(tiff_path)
frames = []
try:
    while True:
        frame = np.array(img, dtype=np.float64)
        frames.append(frame)
        img.seek(img.tell() + 1)
except EOFError:
    pass

n_frames = len(frames)
print(f"Loaded {n_frames} frames")

h, w = frames[0].shape[:2]

# --- FPS handling ---
fps = args.fps
total_duration = n_frames / fps
print(f"Frame size: {w}x{h}, FPS: {fps:.2f}, Duration: {total_duration:.2f}s")

# --- 2. Stack and enhance contrast ---
stack = np.array(frames, dtype=np.float64)

if enhance:
    print("Applying percentile contrast enhancement per frame...")
    for i in range(n_frames):
        stack[i] = enhance_mimg(stack[i]) * 255.0
else:
    if stack.max() > 255:
        stack = (stack - stack.min()) / (stack.max() - stack.min()) * 255.0

# --- 3. Flip upside-down ---
if flip_vertical:
    print("Flipping frames upside-down...")
    stack = stack[:, ::-1, ...]

# --- 4. Running average smoothing ---
print(f"Applying running average (window={running_avg_window})...")
smoothed = np.zeros_like(stack)
half_w = running_avg_window // 2
for i in range(n_frames):
    start = max(0, i - half_w)
    end = min(n_frames, i + half_w + 1)
    smoothed[i] = np.mean(stack[start:end], axis=0)

smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

# --- 5. Encode video ---
print(f"Encoding video to: {output_path}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

for i in range(n_frames):
    frame = smoothed[i]

    if frame.ndim == 2:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    t = i / fps

    # Overlay
    if args.overlay == "timestamp":
        time_text = f"{t:.2f} sec"
        cv2.putText(frame_bgr, time_text, timestamp_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2, cv2.LINE_AA)
    elif args.overlay == "distance":
        depth_um = i * args.um_per_frame
        depth_text = f"{depth_um:.0f} \u00b5m"
        cv2.putText(frame_bgr, depth_text, timestamp_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2, cv2.LINE_AA)

    # Airpuff icon
    if not args.no_puff and airpuff_start <= t < airpuff_start + airpuff_duration:
        draw_airpuff_icon(frame_bgr, airpuff_icon_pos[0], airpuff_icon_pos[1], size=30)

    out.write(frame_bgr)

out.release()
print(f"Done! Saved to: {output_path}")
