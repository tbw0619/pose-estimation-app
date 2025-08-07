import streamlit as st
import tempfile
import cv2
import os
import numpy as np

# Streamlit設定
st.set_page_config(
    page_title="姿勢推定アプリ - MediaPipe × Streamlit",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tbw0619/pose-estimation-app',
        'Report a bug': 'https://github.com/tbw0619/pose-estimation-app/issues',
        'About': """
        # 🧍 姿勢推定アプリ
        
        YOLO7スタイルの高精度姿勢推定アプリケーション
        
        **開発者**: tbw0619  
        **技術**: MediaPipe × Streamlit  
        **GitHub**: https://github.com/tbw0619/pose-estimation-app
        """
    }
)

# タイトル
st.title("🧍 姿勢推定アプリ - MediaPipe × Streamlit")

# サイドバーで設定オプション
st.sidebar.header("⚙️ 設定オプション")

# 解像度設定
resolution_option = st.sidebar.selectbox(
    "処理解像度",
    ["元の解像度を保持", "HD (1280x720)", "Full HD (1920x1080)", "低解像度 (640x480)"],
    index=0
)

# 精度設定
model_complexity = st.sidebar.slider("モデル複雑度 (高いほど正確)", 0, 2, 2)
detection_confidence = st.sidebar.slider("検出信頼度", 0.0, 1.0, 0.7, 0.05)
tracking_confidence = st.sidebar.slider("追跡信頼度", 0.0, 1.0, 0.5, 0.05)

# 描画設定
draw_landmarks = st.sidebar.checkbox("関節点を描画", True)
draw_connections = st.sidebar.checkbox("骨格線を描画", True)
draw_face = st.sidebar.checkbox("首を描画", True, help="肩関節とこめかみを結んで首を表現")
draw_hands = st.sidebar.checkbox("手を描画", True)
landmark_size = st.sidebar.slider("関節点サイズ", 1, 10, 3)
connection_thickness = st.sidebar.slider("骨格線の太さ", 1, 10, 2)

try:
    import mediapipe as mp
    # MediaPipe 初期化
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh  # 顔メッシュ（オプション）
    mp_hands = mp.solutions.hands  # 手の検出追加
    mediapipe_available = True
    st.success("✅ MediaPipe が正常に読み込まれました（YOLO7スタイル姿勢推定）")
except ImportError as e:
    st.error(f"❌ MediaPipe の読み込みに失敗しました: {e}")
    mediapipe_available = False
except Exception as e:
    st.error(f"❌ MediaPipe の初期化中にエラーが発生しました: {e}")
    mediapipe_available = False

# ファイルアップロード設定の説明
st.info("📝 対応ファイル形式: MP4, MOV, AVI（サイズ制限なし - 大きなファイルは処理に時間がかかります）")

# 動画アップローダー
uploaded_file = st.file_uploader(
    "動画ファイルをアップロード", 
    type=["mp4", "mov", "avi"],
    help="動画ファイルを選択してください。大きなファイルでも処理できますが、時間がかかる場合があります。"
)

def get_target_resolution(original_width, original_height, resolution_option):
    """解像度設定に基づいて目標解像度を計算"""
    if resolution_option == "元の解像度を保持":
        return original_width, original_height
    elif resolution_option == "HD (1280x720)":
        target_width, target_height = 1280, 720
    elif resolution_option == "Full HD (1920x1080)":
        target_width, target_height = 1920, 1080
    elif resolution_option == "低解像度 (640x480)":
        target_width, target_height = 640, 480
    else:
        return original_width, original_height
    
    # アスペクト比を維持してリサイズ
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        # 幅に合わせてリサイズ
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # 高さに合わせてリサイズ
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    return new_width, new_height

def draw_pose_landmarks(frame, pose_results, face_results, hands_results, mp_pose, mp_face_mesh, mp_hands, mp_drawing, 
                       draw_landmarks, draw_connections, draw_face, draw_hands, landmark_size, connection_thickness):
    """姿勢ランドマークを描画（YOLO7スタイルの首描画）"""
    
    # 姿勢ランドマーク描画
    if pose_results.pose_landmarks:
        # ランドマーク描画設定
        landmark_drawing_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # 緑色の関節点
            thickness=landmark_size,
            circle_radius=landmark_size
        )
        
        connection_drawing_spec = mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # 赤色の骨格線
            thickness=connection_thickness
        )
        
        # 描画実行
        if draw_landmarks and draw_connections:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )
        elif draw_landmarks:
            # 関節点のみ描画
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, None,
                landmark_drawing_spec=landmark_drawing_spec
            )
        elif draw_connections:
            # 骨格線のみ描画
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0),
                connection_drawing_spec=connection_drawing_spec
            )
        
        # YOLO7スタイルの首描画
        if draw_face:
            # 肩の座標を取得
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # 鼻の座標を取得（こめかみの代わりに使用、より安定）
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            
            # 左右の耳の座標も取得（こめかみにより近い位置）
            left_ear = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            
            h, w, _ = frame.shape
            
            # 首の中心計算（肩の中点）
            neck_center_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # 頭部の中心計算（鼻と耳の平均）
            head_center_x = (nose.x + left_ear.x + right_ear.x) / 3
            head_center_y = (nose.y + left_ear.y + right_ear.y) / 3
            
            # YOLO7スタイル：肩から頭部への首線
            cv2.line(frame,
                    (int(neck_center_x * w), int(neck_center_y * h)),
                    (int(head_center_x * w), int(head_center_y * h)),
                    (0, 255, 255), connection_thickness * 2)  # 黄色の首線
            
            # 左肩から左こめかみ（左耳）
            cv2.line(frame,
                    (int(left_shoulder.x * w), int(left_shoulder.y * h)),
                    (int(left_ear.x * w), int(left_ear.y * h)),
                    (0, 200, 255), connection_thickness)  # オレンジ色
            
            # 右肩から右こめかみ（右耳）
            cv2.line(frame,
                    (int(right_shoulder.x * w), int(right_shoulder.y * h)),
                    (int(right_ear.x * w), int(right_ear.y * h)),
                    (0, 200, 255), connection_thickness)  # オレンジ色
            
            # 首の関節点を描画
            cv2.circle(frame,
                      (int(neck_center_x * w), int(neck_center_y * h)),
                      landmark_size * 2, (0, 255, 255), -1)  # 黄色の首関節
            
            # 頭部中心点を描画
            cv2.circle(frame,
                      (int(head_center_x * w), int(head_center_y * h)),
                      landmark_size, (255, 255, 0), -1)  # 青緑色の頭部中心
    
    # 顔の輪郭描画（オプション、顔メッシュが利用可能な場合のみ）
    if draw_face and face_results and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # 顔の主要な輪郭のみ描画（処理を軽くする）
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=0),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
            )
    
    # 手の描画
    if draw_hands and hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=landmark_size, circle_radius=landmark_size),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=connection_thickness)
            )
    
    return frame

if uploaded_file is not None and mediapipe_available:
    try:
        # ファイルサイズ表示（制限なし）
        file_size = len(uploaded_file.getvalue())
        st.write(f"📁 ファイルサイズ: {file_size / (1024*1024):.1f} MB")
        
        # プログレスバー表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 動画ファイルを処理中...")
        progress_bar.progress(5)
        
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.getvalue())
            video_path = tfile.name
        
        progress_bar.progress(10)
        status_text.text("📹 動画情報を取得中...")
        
        # 動画読み込み
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ 動画ファイルを開けませんでした。ファイル形式を確認してください。")
            os.unlink(video_path)
            st.stop()
        
        # 動画情報取得
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # 目標解像度計算
        target_width, target_height = get_target_resolution(original_width, original_height, resolution_option)
        
        # 動画情報表示
        col1, col2 = st.columns(2)
        with col1:
            st.write("📊 **元の動画情報:**")
            st.write(f"- 解像度: {original_width} × {original_height}")
            st.write(f"- フレーム数: {total_frames}")
            st.write(f"- FPS: {fps:.1f}")
            st.write(f"- 時間: {duration:.1f} 秒")
        
        with col2:
            st.write("⚙️ **処理設定:**")
            st.write(f"- 処理解像度: {target_width} × {target_height}")
            st.write(f"- モデル複雑度: {model_complexity}")
            st.write(f"- 検出信頼度: {detection_confidence}")
            st.write(f"- 追跡信頼度: {tracking_confidence}")
        
        progress_bar.progress(20)
        status_text.text("🤖 MediaPipe初期化中...")
        
        # 動画表示エリア
        video_placeholder = st.empty()
        
        # MediaPipe設定（姿勢中心で軽量化）
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,  # ユーザー設定
            smooth_landmarks=True,  # ランドマークの平滑化
            enable_segmentation=False,  # セグメンテーション無効化（軽量化）
            smooth_segmentation=False,  # セグメンテーション平滑化無効
            min_detection_confidence=detection_confidence,  # ユーザー設定
            min_tracking_confidence=tracking_confidence  # ユーザー設定
        ) as pose, \
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # 両手検出
            min_detection_confidence=detection_confidence * 0.8,  # 手の検出は少し緩く
            min_tracking_confidence=tracking_confidence * 0.8
        ) as hands:
            
            progress_bar.progress(30)
            status_text.text("🏃 姿勢推定処理中...")
            
            frame_count = 0
            processing_times = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # フレーム処理開始時間
                import time
                start_time = time.time()
                
                # フレームリサイズ（アスペクト比維持）
                if (target_width, target_height) != (original_width, original_height):
                    frame = cv2.resize(frame, (target_width, target_height))
                
                # RGBに変換（MediaPipe用）
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 姿勢と手の推定実行（顔メッシュは使用しない）
                pose_results = pose.process(rgb)
                hands_results = hands.process(rgb)
                
                # YOLO7スタイルの描画（顔メッシュなし）
                frame = draw_pose_landmarks(
                    frame, pose_results, None, hands_results,
                    mp_pose, mp_face_mesh, mp_hands, mp_drawing, 
                    draw_landmarks, draw_connections, draw_face, draw_hands,
                    landmark_size, connection_thickness
                )
                
                # 処理時間計算
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # 情報オーバーレイ
                info_text = f"Frame: {frame_count}/{total_frames} | Processing: {processing_time*1000:.1f}ms"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 検出状態表示
                detection_status = []
                if pose_results.pose_landmarks:
                    detection_status.append("POSE")
                if hands_results.multi_hand_landmarks:
                    detection_status.append(f"HANDS({len(hands_results.multi_hand_landmarks)})")
                
                if detection_status:
                    status_text_display = "DETECTED: " + " + ".join(detection_status)
                    cv2.putText(frame, status_text_display, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "NO DETECTION", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # フレーム表示
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # プログレス更新（10フレームごと）
                if frame_count % max(1, total_frames // 50) == 0:
                    progress = 30 + int((frame_count / total_frames) * 60)
                    progress_bar.progress(min(progress, 90))
                    
                    # 推定処理速度表示
                    avg_processing_time = np.mean(processing_times[-10:]) * 1000
                    status_text.text(f"🏃 処理中... (平均: {avg_processing_time:.1f}ms/frame)")
        
        cap.release()
        
        # 処理完了
        progress_bar.progress(100)
        avg_processing_time = np.mean(processing_times) * 1000
        status_text.text(f"✅ 処理完了！平均処理時間: {avg_processing_time:.1f}ms/frame")
        
        # 統計情報表示
        st.success("🎉 姿勢推定が完了しました！")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("処理フレーム数", f"{frame_count}")
        with col2:
            st.metric("平均処理時間", f"{avg_processing_time:.1f}ms")
        with col3:
            estimated_fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
            st.metric("推定リアルタイム性能", f"{estimated_fps:.1f} FPS")
        
        # 一時ファイルを削除
        try:
            os.unlink(video_path)
        except:
            pass
        
    except Exception as e:
        st.error(f"❌ 動画処理中にエラーが発生しました: {e}")
        st.error(f"エラータイプ: {type(e).__name__}")
        import traceback
        st.text("詳細なエラー情報:")
        st.code(traceback.format_exc())
        
elif uploaded_file is not None and not mediapipe_available:
    st.warning("⚠️ MediaPipeが利用できないため、姿勢推定を実行できません。")
else:
    st.info("👆 上記から動画ファイルを選択してください。")
