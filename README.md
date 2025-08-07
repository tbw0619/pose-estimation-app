# 🧍 姿勢推定アプリ - MediaPipe × Streamlit

YOLO7スタイルの高精度姿勢推定アプリケーション。MediaPipeを使用して動画から人体の姿勢を検出し、首・肩・手の詳細な追跡を行います。

## 🚀 特徴

### 📊 高精度な姿勢推定
- **YOLO7スタイル首描画**: 肩関節とこめかみを結んだ確実な首の表現
- **33点関節検出**: MediaPipeによる詳細な人体ランドマーク
- **両手追跡**: 21点の手指関節追跡
- **リアルタイム処理**: 高速な動画解析

### 🎛️ カスタマイズ可能な設定
- **解像度選択**: 元解像度〜4K対応
- **精度調整**: モデル複雑度・信頼度の細かい設定
- **描画オプション**: 関節点・骨格線・首・手の個別ON/OFF
- **視覚設定**: サイズ・太さの調整

### 📁 ファイルサポート
- **対応形式**: MP4, MOV, AVI
- **サイズ制限なし**: 大容量動画も処理可能
- **進捗表示**: リアルタイム処理状況

## 🖥️ デモ

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](あなたのアプリURL)

## 🛠️ 技術スタック

- **Frontend**: Streamlit
- **AI/ML**: MediaPipe
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy
- **Deployment**: Streamlit Community Cloud

## 📦 インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/pose-estimation-app.git
cd pose-estimation-app

# 依存関係をインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

## 🎯 使用方法

1. **動画アップロード**: MP4/MOV/AVIファイルを選択
2. **設定調整**: サイドバーで精度・描画オプションを設定
3. **処理実行**: 自動で姿勢推定が開始
4. **結果確認**: リアルタイムで結果を表示

### 推奨設定
- **モデル複雑度**: 2（最高精度）
- **検出信頼度**: 0.7
- **追跡信頼度**: 0.5

## 🎨 描画オプション

- 🟢 **関節点**: 緑色の人体関節
- 🔴 **骨格線**: 赤色の骨格構造
- 🟡 **首**: YOLO7スタイルの黄色い首線
- 🔵 **手**: シアン色の手指関節

## 📈 パフォーマンス

- **処理速度**: 平均20-60ms/frame
- **メモリ使用量**: 最適化済み
- **対応解像度**: 640×480 〜 1920×1080+

## 🤝 貢献

Issue報告やPull Requestを歓迎します！

## 📄 ライセンス

MIT License

## 🔗 関連リンク

- [MediaPipe](https://mediapipe.dev/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
