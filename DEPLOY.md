# 🚀 デプロイガイド

## Streamlit Community Cloudでのデプロイ

### 1. GitHubリポジトリの準備 ✅
- [x] GitHubリポジトリ作成完了
- [x] 必要ファイル準備完了
  - `app.py` - メインアプリケーション
  - `requirements.txt` - 依存関係
  - `.streamlit/config.toml` - Streamlit設定
  - `README.md` - プロジェクト説明

### 2. Streamlit Community Cloudでデプロイ

1. **https://share.streamlit.io** にアクセス
2. **GitHubアカウントでサインイン**
3. **"New app"をクリック**
4. **リポジトリ設定**:
   - Repository: `tbw0619/pose-estimation-app`
   - Branch: `main`
   - Main file path: `app.py`
5. **"Deploy!"をクリック**

### 3. デプロイ後の確認

- ✅ アプリケーションが正常に起動
- ✅ MediaPipeライブラリの読み込み
- ✅ ファイルアップロード機能
- ✅ 姿勢推定処理

### 4. カスタムドメイン（オプション）

デプロイ完了後、Streamlit Community Cloudのダッシュボードから:
1. アプリ設定を開く
2. "Settings" → "General" → "App URL"
3. カスタムURLを設定可能

## 🔧 本番環境での最適化

### パフォーマンス設定
- ファイルサイズ制限: 500MB
- 処理速度最適化: セグメンテーション無効化
- メモリ使用量最適化: OpenCV headless版使用

### セキュリティ設定
- CORS設定: 適切に構成済み
- ファイルアップロード: 許可された形式のみ
- エラーハンドリング: 詳細なエラー情報表示

## 📊 監視とメンテナンス

### ログ確認
- Streamlit Community Cloud管理画面でログ確認
- エラー発生時の自動通知

### アップデート
```bash
git add .
git commit -m "Update: 説明"
git push
```
→ 自動で再デプロイ

## 🔗 デプロイ後のURL

デプロイ完了後のアプリURL:
**https://pose-estimation-app-tbw0619.streamlit.app/**
(実際のURLはデプロイ時に確定)

## 📞 サポート

問題が発生した場合:
1. GitHubのIssueで報告
2. Streamlit Community Cloudのサポート
3. 開発者への直接連絡
