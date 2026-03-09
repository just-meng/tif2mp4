#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "opencv-python",
# ]
# ///

import numpy as np
import cv2
import argparse

# ============== ARGUMENT PARSING ==============
parser = argparse.ArgumentParser(description="Re-encode an .mp4 with new duration, timestamp, and airpuff overlay.")
parser.add_argument("input_path", help="Path to the input .mp4 file")
parser.add_argument("output_path", help="Path for the output .mp4 file")
parser.add_argument("--target-duration", type=float, default=10.0, help="Target video duration in seconds (default: 10.0)")
parser.add_argument("--puff-start", type=float, default=2.0, help="Airpuff onset time in seconds (default: 2.0)")
parser.add_argument("--puff-duration", type=float, default=1.0, help="Airpuff display duration in seconds (default: 1.0)")
parser.add_argument("--no-puff", action="store_true", help="Disable airpuff icon overlay")
parser.add_argument("--overlay", choices=["timestamp", "distance", "none"], default="timestamp",
                    help="Overlay type: 'timestamp', 'distance', or 'none' (default: timestamp)")
parser.add_argument("--um-per-frame", type=float, default=5.0, help="Microns per frame for distance overlay (default: 5.0)")
parser.add_argument("--no-enhance", action="store_true", help="Disable percentile contrast enhancement")
args = parser.parse_args()

# ============== OVERLAY SETTINGS ==============
font_scale = 3.0
font_thickness = 6
font_color = (255, 255, 255)
airpuff_icon_size = 80

# ============== HELPER FUNCTIONS ==============
def enhance_frame(frame):
    """Percentile-clip and normalise a frame to 0-255 uint8."""
    f = frame.astype(np.float64)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)
    return np.clip((f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

def get_text_top_right(text, frame_w, font, font_scale, thickness, margin=20):
    """Return position to place text in top-right corner."""
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = frame_w - tw - margin
    y = th + margin
    return (x, y)

def draw_airpuff_icon(img_bgr, x, y, size=80):
    """Draws a nozzle pointing LEFT (into the video) with air burst lines and 'AIR PUFF' label."""
    # Nozzle triangle pointing left
    nozzle_pts = np.array([
        [x, y],                          # tip (left)
        [x - size // 3, y - size // 4],  # top-right of nozzle
        [x - size // 3, y + size // 4]   # bottom-right of nozzle
    ], dtype=np.int32)
    cv2.fillPoly(img_bgr, [nozzle_pts], (0, 200, 255))

    # Air burst lines going left from tip
    line_start_x = x - 2
    for dy in [-size // 4, -size // 8, 0, size // 8, size // 4]:
        start_pt = (line_start_x, y + dy)
        end_pt = (line_start_x - size // 2, y + dy + (dy // 2))
        cv2.line(img_bgr, start_pt, end_pt, (0, 200, 255), 3, cv2.LINE_AA)

    # Label to the right of nozzle
    cv2.putText(img_bgr, "AIR PUFF", (x - size // 3 + 10, y + size // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3, cv2.LINE_AA)
    return img_bgr

# --- 1. Read input video ---
print(f"Reading input video: {args.input_path}")
cap = cv2.VideoCapture(args.input_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.input_path}")

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_duration = n_frames / original_fps

print(f"Frames: {n_frames}, Original FPS: {original_fps:.2f}, Original duration: {original_duration:.2f}s")
print(f"Frame size: {w}x{h}")

# --- 2. Compute new FPS ---
new_fps = n_frames / args.target_duration
print(f"New FPS: {new_fps:.2f} (to fit {n_frames} frames in {args.target_duration}s)")

# --- 3. Read all frames ---
print("Reading frames...")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Read {len(frames)} frames")

# --- 4. Airpuff icon position (top-right area, below where timestamp goes) ---
puff_x = w - 60
puff_y = 120

# --- 5. Encode output video ---
print(f"Encoding video to: {args.output_path}")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_path, fourcc, new_fps, (w, h), isColor=True)

for i, frame_bgr in enumerate(frames):

    # Enhance
    if not args.no_enhance:
        frame_bgr = enhance_frame(frame_bgr)

    t = i / new_fps

    # Overlay (top-right)
    if args.overlay == "timestamp":
        time_text = f"{t:.2f} sec"
        pos = get_text_top_right(time_text, w, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(frame_bgr, time_text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
    elif args.overlay == "distance":
        depth_um = i * args.um_per_frame
        depth_text = f"{depth_um:.0f} um"
        pos = get_text_top_right(depth_text, w, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(frame_bgr, depth_text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Airpuff icon (top-right, below text, pointing into video)
    if not args.no_puff and args.puff_start <= t < args.puff_start + args.puff_duration:
        draw_airpuff_icon(frame_bgr, puff_x, puff_y, size=airpuff_icon_size)

    out.write(frame_bgr)

out.release()
print(f"Done! {n_frames} frames @ {new_fps:.2f} FPS = {args.target_duration:.1f}s")
print(f"Saved to: {args.output_path}")
