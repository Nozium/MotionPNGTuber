# ONNX Models for Anime Face Detection

このディレクトリには anime-face-detector の ONNX 変換済みモデルを配置します。

## モデル構成

| ファイル | 説明 | 入力サイズ | 出力 |
|---------|------|-----------|------|
| `anime_face_yolov3.onnx` | 顔検出 (YOLOv3) | 608x608 | bbox [x1,y1,x2,y2,conf] |
| `anime_landmark_hrnetv2.onnx` | ランドマーク検出 (HRNetV2) | 256x256 | 28点 keypoints |

## ONNX モデルの生成方法

### 1. 依存パッケージのインストール

```bash
# OpenMIM (MM系ライブラリ管理ツール)
pip install openmim

# MM系ライブラリ
mim install mmcv-full mmdet mmpose

# anime-face-detector と ONNX
pip install anime-face-detector onnx onnxruntime

# (オプション) ONNX 最適化
pip install onnxsim
```

### 2. エクスポートスクリプトの実行

```bash
python scripts/export_onnx.py --output-dir models/onnx
```

### 3. 出力ファイル

```
models/onnx/
├── anime_face_yolov3.onnx      # ~60MB
├── anime_landmark_hrnetv2.onnx # ~40MB
└── export_info.json            # エクスポート情報
```

## ONNX モデルの使用方法

### 軽量環境での使用

MM系ライブラリなしで動作可能:

```bash
pip install onnxruntime numpy opencv-python
# GPU使用時: pip install onnxruntime-gpu
```

### コード例

```python
from scripts.anime_face_detector_onnx import create_detector
import cv2

# 検出器を作成
detector = create_detector(device="cpu", onnx_dir="models/onnx")

# 画像を読み込んで検出
image = cv2.imread("anime_face.png")
results = detector(image)

for r in results:
    print(f"BBox: {r['bbox']}")
    print(f"Keypoints shape: {r['keypoints'].shape}")  # (28, 3)
```

### anime-face-detector からの移行

```python
# Before (mm系依存)
from anime_face_detector import create_detector
detector = create_detector("yolov3", device="cuda:0")

# After (ONNX版)
from scripts.anime_face_detector_onnx import create_detector
detector = create_detector("yolov3", device="cuda:0", onnx_dir="models/onnx")
```

## キーポイントのインデックス

| Index | 部位 |
|-------|------|
| 0-10 | 顔輪郭 |
| 11-16 | 左目 |
| 17-22 | 右目 |
| 23 | 鼻 |
| 24-27 | 口 (MotionPNGTuber で使用) |

## 注意事項

- YOLOv3 + HRNetV2 の合計サイズは約 100MB です
- Git LFS を使用してコミットすることを推奨します
- モデルの再配布は元のライセンス (anime-face-detector) に従ってください

## ライセンス

元のモデル: [anime-face-detector](https://github.com/hysts/anime-face-detector)
- 顔検出: YOLOv3 trained on anime faces
- ランドマーク: HRNetV2 (28 keypoints)
