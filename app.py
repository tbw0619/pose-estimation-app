import streamlit as st
import os
import sys
import tempfile
import time
import cv2
import numpy as np
import imageio
import mediapipe as mp

# ---- Streamlit page config ----
st.set_page_config(page_title="Pose Estimation (GIF, 15s)", page_icon="🧍", layout="wide")

st.title("🧍 Pose Estimation (15s GIF) — MediaPipe × Streamlit")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
model_complexity = st.sidebar.slider("Model complexity (higher = more accurate)", 0, 2, 2)
min_det_conf = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.7, 0.05)
min_track_conf = st.sidebar.slider("Min tracking confidence", 0.0, 1.0, 0.7, 0.05)
max_seconds = st.sidebar.slider("Process duration (seconds)", 3, 30, 15, 1)
max_gif_width = st.sidebar.selectbox("Max GIF width", [480, 640, 720], index=2)
draw_hands = st.sidebar.checkbox("Draw hands", False)
draw_neck = st.sidebar.checkbox("Draw neck helper line", True, help="Approximate neck line between shoulder-center and head center")

st.markdown("**Upload a video (mp4/mov/avi). We'll process up to the first 15 seconds and return an annotated GIF.**")

uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi"])
if not uploaded:
    st.stop()

# Save to temp file so OpenCV can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
    t.write(uploaded.read())
    in_path = t.name

# Read video metadata
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

# We'll sample to ~10 FPS for GIF to keep size reasonable
gif_fps = 10
sample_every = max(1, int(round(fps / gif_fps)))
frames_to_process = int(min(duration, max_seconds) * fps)

progress = st.progress(0)
status = st.empty()

# Init MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
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

hands = None
if draw_hands:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf
    )

def draw_annotations(frame_bgr, pose_result, hands_result):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Pose
    if pose_result and pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame_bgr,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2),
        )
        if draw_neck:
            lm = pose_result.pose_landmarks.landmark
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = lm[mp_pose.PoseLandmark.NOSE]
            le = lm[mp_pose.PoseLandmark.LEFT_EAR]
            re = lm[mp_pose.PoseLandmark.RIGHT_EAR]
            nx = (ls.x + rs.x) / 2.0
            ny = (ls.y + rs.y) / 2.0
            hx = (nose.x + le.x + re.x) / 3.0
            hy = (nose.y + le.y + re.y) / 3.0
            cv2.line(frame_bgr, (int(nx*w), int(ny*h)), (int(hx*w), int(hy*h)), (0,255,255), 3)
            cv2.circle(frame_bgr, (int(nx*w), int(ny*h)), 4, (0,255,255), -1)
            cv2.circle(frame_bgr, (int(hx*w), int(hy*h)), 3, (255,255,0), -1)

    # Hands (optional)
    if hands_result and hands_result.multi_hand_landmarks:
        for hand_lm in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr, hand_lm, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,255), thickness=2),
            )
    return frame_bgr

# Collect frames for GIF
gif_frames = []
frame_idx = 0
sampled = 0
last_update = time.time()

while frame_idx < frames_to_process:
    ok, frame = cap.read()
    if not ok:
        break
    # Downscale for speed if width is too large (target <= 720 px)
    scale = min(1.0, max_gif_width / max(orig_w, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(orig_w*scale), int(orig_h*scale)))

    # Process only every Nth frame for GIF FPS
    if frame_idx % sample_every == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)

        hands_res = None
        if hands is not None:
            hands_res = hands.process(rgb)

        annotated = draw_annotations(frame.copy(), pose_res, hands_res)
        gif_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        sampled += 1

        # Update UI occasionally
        if time.time() - last_update > 0.1:
            status.text(f"Processing… {frame_idx}/{frames_to_process} frames (kept {sampled} for GIF)")
            progress.progress(min(100, int(100 * frame_idx / max(1, frames_to_process))))
            last_update = time.time()

    frame_idx += 1

cap.release()
pose.close()
if hands is not None:
    hands.close()

if not gif_frames:
    st.error("No frames were processed. Try a different video.")
    st.stop()

# Save GIF
gif_path = os.path.join(tempfile.gettempdir(), "pose_annotated.gif")
# duration per frame in seconds
frame_duration = 1.0 / gif_fps
imageio.mimsave(gif_path, gif_frames, format="GIF", duration=frame_duration)

st.success(f"Done. {sampled} frames -> GIF at ~{gif_fps} FPS")
st.image(gif_path, caption="Annotated 15s GIF (preview)", use_column_width=True)

with open(gif_path, "rb") as f:
    st.download_button("Download GIF", f, file_name="pose_annotated.gif", mime="image/gif")
