# --- MUST be the very first lines ---
import os, tempfile
mp_cache = os.path.join(tempfile.gettempdir(), "mediapipe_cache")
os.environ["XDG_CACHE_HOME"] = mp_cache
os.environ["MEDIAPIPE_CACHE_DIR"] = mp_cache
os.environ["HOME"] = tempfile.gettempdir()
os.makedirs(mp_cache, exist_ok=True)
# ------------------------------------

import streamlit as st
import cv2
import numpy as np
import imageio
import time

# ここより下で初めて mediapipe を import
import mediapipe as mp


st.set_page_config(page_title="Pose Estimation (GIF, 15s)", page_icon="🧍", layout="wide")
st.title("🧍 Pose Estimation (15s GIF) — MediaPipe × Streamlit (Cache Fix)")

st.sidebar.header("⚙️ Settings")
model_complexity = st.sidebar.slider("Model complexity (higher = more accurate)", 0, 2, 2)
min_det_conf = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.7, 0.05)
min_track_conf = st.sidebar.slider("Min tracking confidence", 0.0, 1.0, 0.7, 0.05)
max_seconds = st.sidebar.slider("Process duration (seconds)", 3, 30, 15, 1)
max_gif_width = st.sidebar.selectbox("Max GIF width", [480, 640, 720], index=2)

st.markdown("**Upload a video (mp4/mov/avi). We'll process up to the first 15 seconds and return an annotated GIF.**")

uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi"])
if not uploaded:
    st.stop()

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

gif_fps = 10
sample_every = max(1, int(round(fps / gif_fps)))
frames_to_process = int(min(duration, max_seconds) * fps)

progress = st.progress(0)
status = st.empty()

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=model_complexity,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=min_det_conf,
    min_tracking_confidence=min_track_conf
)

def draw_annotations(frame_bgr, pose_result):
    if pose_result and pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame_bgr,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(thickness=2, circle_radius=2),
            connection_drawing_spec=mp_draw.DrawingSpec(thickness=2),
        )
    return frame_bgr

gif_frames = []
frame_idx = 0
sampled = 0
last_update = time.time()

while frame_idx < frames_to_process:
    ok, frame = cap.read()
    if not ok:
        break
    scale = min(1.0, max_gif_width / max(orig_w, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(orig_w*scale), int(orig_h*scale)))
    if frame_idx % sample_every == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        annotated = draw_annotations(frame.copy(), pose_res)
        gif_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        sampled += 1
        if time.time() - last_update > 0.1:
            status.text(f"Processing… {frame_idx}/{frames_to_process} frames (kept {sampled} for GIF)")
            progress.progress(min(100, int(100 * frame_idx / max(1, frames_to_process))))
            last_update = time.time()
    frame_idx += 1

cap.release()
pose.close()

if not gif_frames:
    st.error("No frames were processed. Try a different video.")
    st.stop()

gif_path = os.path.join(tempfile.gettempdir(), "pose_annotated.gif")
imageio.mimsave(gif_path, gif_frames, format="GIF", duration=1.0/gif_fps)

st.success(f"Done. {sampled} frames -> GIF at ~{gif_fps} FPS (cached at {mp_cache})")
st.image(gif_path, caption="Annotated 15s GIF (preview)", use_column_width=True)
with open(gif_path, "rb") as f:
    st.download_button("Download GIF", f, file_name="pose_annotated.gif", mime="image/gif")
