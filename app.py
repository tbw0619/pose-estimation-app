import os, tempfile, urllib.request, pathlib
import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- キャッシュ/tmp設定 ---
os.environ["HOME"] = tempfile.gettempdir()
os.environ["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

st.set_page_config(page_title="Pose Estimation (Tasks, 15s MP4)", page_icon="🧍", layout="wide")
st.title("🧍 Pose Estimation — MediaPipe Tasks (15s MP4出力)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_seconds = st.sidebar.slider("Process duration (seconds)", 3, 30, 15, 1)
max_video_width = st.sidebar.selectbox("Max video width", [480, 640, 720], index=2)
video_fps = st.sidebar.slider("Output video FPS", 5, 30, 10, 1)

st.markdown("**Upload a video (mp4/mov/avi). We'll process up to the first 15 seconds and return an annotated MP4 video.**")

# --- モデル確保 ---
DEFAULT_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
repo_model_path = pathlib.Path("models/pose_landmarker_heavy.task")
tmp_model_path = pathlib.Path(tempfile.gettempdir()) / "pose_landmarker_heavy.task"

def ensure_model() -> str:
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

sample_every = max(1, int(round(fps / video_fps)))
frames_to_process = int(min(duration, max_seconds) * fps)

progress = st.progress(0)
status = st.empty()

def draw_pose(frame_bgr, landmarks_list):
    h, w = frame_bgr.shape[:2]
    if not landmarks_list:
        return frame_bgr
    lm = landmarks_list[0]
    CONN = [
        (11,13),(13,15),
        (12,14),(14,16),
        (11,12),
        (23,24),
        (11,23),(12,24),
        (23,25),(25,27),(27,29),(29,31),
        (24,26),(26,28),(28,30),(30,32),
    ]
    pts = []
    for p in lm:
        x = int(p.x * w); y = int(p.y * h)
        pts.append((x,y))
        cv2.circle(frame_bgr,(x,y),2,(0,255,0),-1)
    for a,b in CONN:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], (255,0,0), 2)
    return frame_bgr

# --- 出力動画設定 ---
video_path = os.path.join(tempfile.gettempdir(), "pose_tasks.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_w = int(orig_w * min(1.0, max_video_width / max(orig_w, 1)))
out_h = int(orig_h * min(1.0, max_video_width / max(orig_w, 1)))
out = cv2.VideoWriter(video_path, fourcc, video_fps, (out_w, out_h))

frame_idx = 0
kept = 0
last_update = time.time()
timestamp_ms = 0.0

while frame_idx < frames_to_process:
    ok, frame = cap.read()
    if not ok:
        break
    if max_video_width < orig_w:
        frame = cv2.resize(frame, (out_w, out_h))

    if frame_idx % sample_every == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        annotated = draw_pose(frame.copy(), result.pose_landmarks)
        out.write(annotated)
        kept += 1
        if time.time() - last_update > 0.1:
            status.text(f"Processing… {frame_idx}/{frames_to_process} frames (written {kept})")
            progress.progress(min(100, int(100 * frame_idx / max(1, frames_to_process))))
            last_update = time.time()
    frame_idx += 1
    timestamp_ms += 1000.0 / fps

cap.release()
out.release()

if kept == 0:
    st.error("No frames were processed. Try a different video.")
    st.stop()

# --- 再生とダウンロード ---
st.success(f"Done. {kept} frames -> MP4 at ~{video_fps} FPS")
st.video(video_path)
with open(video_path, "rb") as f:
    st.download_button(
        "Download MP4", 
        f, 
        file_name="pose_annotated.mp4", 
        mime="video/mp4"
    )
