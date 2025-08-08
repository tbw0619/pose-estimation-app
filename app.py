
import os, tempfile, urllib.request, pathlib
import streamlit as st
import cv2
import numpy as np
import imageio
import time

# --- Configure cache/tmp dirs early
os.environ["HOME"] = tempfile.gettempdir()
os.environ["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

# --- MediaPipe Tasks (no solutions.Pose to avoid package write) ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="Pose Estimation (Tasks, 15s GIF)", page_icon="🧍", layout="wide")
st.title("🧍 Pose Estimation — MediaPipe Tasks (15s GIF)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_seconds = st.sidebar.slider("Process duration (seconds)", 3, 30, 15, 1)
max_gif_width = st.sidebar.selectbox("Max GIF width", [480, 640, 720], index=2)
gif_fps = st.sidebar.slider("GIF FPS", 5, 15, 10, 1)

st.markdown("**Upload a video (mp4/mov/avi). We'll process up to the first 15 seconds and return an annotated GIF.**")

# --- Ensure model (.task) ---
DEFAULT_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
repo_model_path = pathlib.Path("models/pose_landmarker_heavy.task")
tmp_model_path = pathlib.Path(tempfile.gettempdir()) / "pose_landmarker_heavy.task"

def ensure_model() -> str:
    # priority: repo ./models > /tmp (download) 
    if repo_model_path.exists():
        return str(repo_model_path)
    if not tmp_model_path.exists():
        try:
            st.info("Downloading pose model to temporary directory…")
            urllib.request.urlretrieve(DEFAULT_TASK_URL, tmp_model_path)
        except Exception as e:
            st.error("Model download failed. Please add models/pose_landmarker_heavy.task to the repo.")
            raise
    return str(tmp_model_path)

model_path = ensure_model()

# Build landmarker
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=1
)
landmarker = vision.PoseLandmarker.create_from_options(options)

uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi"])
if not uploaded:
    st.stop()

# Save upload to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
    t.write(uploaded.read())
    in_path = t.name

cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    st.error("Could not open the video. Please try another file.")
    st.stop()

orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = nframes / fps if fps > 0 else 0
st.write(f"**Input:** {orig_w}×{orig_h}, {fps:.1f} FPS, {duration:.1f} s")

sample_every = max(1, int(round(fps / gif_fps)))
frames_to_process = int(min(duration, max_seconds) * fps)

progress = st.progress(0)
status = st.empty()

def draw_pose(frame_bgr, landmarks_list):
    h, w = frame_bgr.shape[:2]
    # Draw simple skeleton lines: if there's at least one pose.
    if not landmarks_list:
        return frame_bgr
    lm = landmarks_list[0]  # first pose
    # List of connections based on Mediapip Pose topology (subset)
    CONN = [
        (11,13),(13,15),  # left arm
        (12,14),(14,16),  # right arm
        (11,12),          # shoulders
        (23,24),          # hips
        (11,23),(12,24),  # torso
        (23,25),(25,27),(27,29),(29,31),  # left leg
        (24,26),(26,28),(28,30),(30,32),  # right leg
    ]
    pts = []
    for i,p in enumerate(lm):
        x = int(p.x * w)
        y = int(p.y * h)
        pts.append((x,y))
        cv2.circle(frame_bgr,(x,y),2,(0,255,0),-1)
    for a,b in CONN:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], (255,0,0), 2)
    return frame_bgr

gif_frames = []
frame_idx = 0
kept = 0
last_update = time.time()

timestamp_ms = 0
while frame_idx < frames_to_process:
    ok, frame = cap.read()
    if not ok:
        break
    scale = min(1.0, max_gif_width / max(orig_w, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(orig_w*scale), int(orig_h*scale)))

    if frame_idx % sample_every == 0:
        # Create an MP Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        annotated = draw_pose(frame.copy(), result.pose_landmarks)
        gif_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        kept += 1
        if time.time() - last_update > 0.1:
            status.text(f"Processing… {frame_idx}/{frames_to_process} frames (kept {kept} for GIF)")
            progress.progress(min(100, int(100 * frame_idx / max(1, frames_to_process))))
            last_update = time.time()
    frame_idx += 1
    timestamp_ms += 1000.0 / fps

cap.release()

if not gif_frames:
    st.error("No frames were processed. Try a different video.")
    st.stop()

gif_path = os.path.join(tempfile.gettempdir(), "pose_tasks.gif")
imageio.mimsave(gif_path, gif_frames, format="GIF", duration=1.0/gif_fps)

st.success(f"Done. {kept} frames -> GIF at ~{gif_fps} FPS")
st.image(gif_path, caption="Annotated 15s GIF (MediaPipe Tasks)", use_column_width=True)
with open(gif_path, "rb") as f:
    st.download_button("Download GIF", f, file_name="pose_annotated.gif", mime="image/gif")
