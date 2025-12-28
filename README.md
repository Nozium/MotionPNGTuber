# MotionPNGTuber

**PNGTuber以上、Live2D未満** — 動画ベースのリアルタイム口パク（リップシンク）システム

ループ動画を使うことで、従来のPNGTuberでは表現できなかった**髪の毛の揺れ**や**衣装のなびき**をリッチに表現できます。Live2Dのような専門知識は不要で、MP4動画と口スプライトさえあれば始められます。

## ✨ 特徴

- 🎤 **リアルタイム口パク**: マイク入力に合わせてキャラクターの口が動く
- 🤖 **AITuber対応**: 仮想オーディオデバイスを使えばAITuberのアバターとしても利用可能
- 🎭 **感情自動判定**: 音声から感情（喜び・怒り・悲しみ等）を推定し、表情を自動切替
- 💨 **髪・揺れ物の動き**: ループ動画なので髪や衣装が自然に揺れる
- 🖼️ **高精度口消し**: 元動画から口を自然に消去し、口スプライトを合成
- 🎯 **簡単キャリブレーション**: マウス操作で口の位置・サイズ・回転を調整
- 🖥️ **GUIツール**: ワンクリックで解析から実行まで
- ⚡ **軽量モード対応**: YOLOv9-Wholebody25による軽量検出（ONNX Runtime、CPU動作可）

## 🎯 こんな人におすすめ

| ユースケース | 説明 |
|--------------|------|
| **PNGTuberからのステップアップ** | 静止画では物足りないけどLive2Dは難しい |
| **VTuber配信** | マイク入力でリアルタイムにアバターを動かしたい |
| **AITuber** | AI音声出力を仮想オーディオ経由でアバターに反映したい |

---

## 🚀 クイックスタート

サンプルアセットを使って、すぐに試すことができます。

### 0. 前提条件（未インストールの場合）

- **Python 3.10**: [ダウンロード](https://www.python.org/downloads/)（インストール時に「Add Python to PATH」にチェック）
- **Visual C++ Build Tools**: [ダウンロード](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（「C++ によるデスクトップ開発」を選択）

### 1. インストール（初回のみ）

#### 🚀 軽量版（推奨・簡単）

```bash
# 仮想環境を作成・有効化
python -m venv .venv
.venv\Scripts\activate

# 依存パッケージをインストール
pip install onnxruntime opencv-python numpy Pillow sounddevice scipy
```

> 💡 軽量版は `--detector yolov9` を使用し、mmcv/mmdet/mmpose のインストールが不要です。

#### 📦 フル版（anime-face-detector使用）

```bash
# 仮想環境を作成・有効化
python -m venv .venv
.venv\Scripts\activate

# PyTorch をインストール（GPU版）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# ※ CPU版の場合はこちら:
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 依存パッケージをインストール
pip install openmim
mim install mmcv-full==1.7.0
pip install mmdet==2.28.0 mmpose==0.29.0 anime-face-detector==0.0.9
pip install opencv-python==4.8.1.78 numpy Pillow sounddevice scipy
```

### 2. GUIを起動

```bash
python mouth_track_gui.py
```

### 3. サンプルで試す

1. **動画を選択**: `assets/assets03/loop.mp4` を選択
2. **mouthフォルダを選択**: `assets/assets03/mouth` を選択
3. **① 解析→キャリブ**: クリックして口の位置を調整 → Space で確定
4. **② 口消し動画生成**: クリックして待つ
5. **③ ライブ実行**: マイクに向かって話すと口が動く！

> 💡 各アセットフォルダには `パラメーター参考例XX.png` が入っています。設定の参考にしてください。

---

## 📦 準備するもの

### 1. アニメキャラクターの動画（.mp4）

- ループ再生できる短い動画（数秒程度）
- **顔が隠れていないもの**（顔が隠れているとリップシンクが正しく動作しません）
- 髪の揺れや衣装のなびきがあるとリッチな表現に

### 2. 口スプライト画像（.png × 5枚）

キャラクターに合った口の画像を5種類用意します：

| ファイル名 | 説明 |
|------------|------|
| `open.png` | 口を開けた状態 |
| `closed.png` | 口を閉じた状態 |
| `half.png` | 半開き |
| `e.png` | 上記3つとは異なる形状（任意） |
| `u.png` | 上記4つとは異なる形状（任意） |

- 画像形式: PNG（透過対応・RGBA）
- 推奨サイズ: 幅128px程度

### 3. 感情別口スプライト（オプション）

感情表現を使いたい場合は、感情ごとに口スプライトを用意します（Default / Happy / Angry / Sad / Excited）。

**感情別スプライトがなくても1種類の口スプライトだけで動作します。** 表現の幅を広げたい場合に追加してください。

---

## 💻 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.10（3.10.x推奨） |
| GPU | NVIDIA GPU + CUDA 11.8（推奨）または CPU のみ |
| RAM | 8GB以上推奨 |

### GPUとCPUについて

- **GPU（CUDA）使用時**: 顔トラッキングが高速（約30FPS）
- **CPUのみ**: トラッキングは低速（約2-5FPS）だが動作可能。リアルタイム再生自体はCPUで十分

### 検出器の比較

| 項目 | YOLOv9（軽量版） | anime-face-detector（フル版） |
|------|------------------|------------------------------|
| **依存パッケージ** | `onnxruntime` のみ (~50MB) | `mmcv-full`, `mmdet`, `mmpose` (~1GB+) |
| **インストール難易度** | 簡単 | やや難しい（mmcv-fullのビルドが必要） |
| **CPU動作** | ✅ 高速 | ⚠️ 低速 |
| **GPU動作** | ✅ 対応 | ✅ 対応（CUDA必須） |
| **検出方式** | 顔/口BBox + 頭部方向 | 28点ランドマーク |
| **コマンドオプション** | `--detector yolov9` | `--detector anime` |

> 💡 **推奨**: まずは軽量版（YOLOv9）をお試しください。インストールが簡単で、CPUでも十分な速度で動作します。

---

## 🔧 インストール手順（Windows）

### 1. 前提条件

#### Microsoft Visual C++ Build Tools

一部のパッケージをコンパイルするために必要です。

1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) をダウンロード
2. インストーラーで「**C++ によるデスクトップ開発**」を選択してインストール

#### Python 3.10

1. [Python 3.10.x](https://www.python.org/downloads/) をダウンロード
2. インストール時に「**Add Python to PATH**」にチェック

#### FFmpeg（オプション・推奨）

口消し動画に音声を残す場合に必要です。

1. [FFmpeg公式サイト](https://ffmpeg.org/download.html) からダウンロード
2. 解凍して `bin` フォルダを環境変数 `PATH` に追加

### 2. 仮想環境の作成

```bash
# プロジェクトディレクトリに移動
cd MotionPNGTuber

# 仮想環境を作成
python -m venv .venv

# 仮想環境を有効化
.venv\Scripts\activate
```

### 3. PyTorchのインストール

**重要**: PyTorchは必ず先にインストールしてください。

#### GPU版（CUDA 11.8）

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### CPU版

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 4. mmcv-fullのインストール

**重要**: `mmcv-full` は pip ではなく `mim` でインストールします。これが最も難しいステップです。

```bash
# openmimをインストール
pip install openmim

# mmcv-fullをインストール（時間がかかります）
mim install mmcv-full==1.7.0
```

**トラブルシューティング**:
- エラーが出る場合は、Visual C++ Build Toolsがインストールされているか確認
- `mim install` が失敗する場合、`pip install mmcv-full==1.7.0` を試す（ただし非推奨）

### 5. その他の依存パッケージ

```bash
# MMDetection と MMPose
pip install mmdet==2.28.0 mmpose==0.29.0

# anime-face-detector
pip install anime-face-detector==0.0.9

# 基本パッケージ
pip install opencv-python==4.8.1.78 numpy Pillow sounddevice scipy

# オプション: OBS仮想カメラ出力
pip install pyvirtualcam
```

### インストール確認

```bash
python -c "import cv2; import numpy; import sounddevice; from anime_face_detector import create_detector; print('OK')"
```

---

## 📁 フォルダ構成

```
MotionPNGTuber/
├── mouth_track_gui.py              # メインGUI
├── mouth_erase_tuner_gui.py        # 口消しチューナーGUI（おまけツール）
├── loop_lipsync_runtime_patched_emotion_auto.py  # リアルタイム実行（感情対応）
├── face_track_anime_detector.py    # 顔トラッキング（anime/yolov9切替対応）
├── yolov9_wholebody_detector.py    # YOLOv9軽量検出器
├── calibrate_mouth_track.py        # キャリブレーション
├── erase_mouth_offline.py          # 口消し処理
├── auto_mouth_track_v2.py          # 自動トラッキング
├── auto_erase_mouth.py             # 自動口消し
├── preview_mouth_track.py          # トラッキングプレビュー
├── realtime_emotion_audio.py       # 感情解析
├── requirements.txt
├── models/                         # ONNXモデル（自動ダウンロード）
│   └── yolov9_wholebody25/
├── assets/                         # 動画アセット（サンプル）
│   ├── assets01/
│   │   ├── loop.mp4
│   │   └── パラメーター参考例01.png
│   ├── assets02/
│   │   ├── loop.mp4
│   │   └── パラメーター参考例02.png
│   └── assets03/
│       ├── loop.mp4
│       ├── mouth/                  # 個別の口スプライト
│       └── パラメーター参考例03.png
└── mouth_dir/                      # 感情別口スプライト
    └── Tomari/
        ├── Default/
        ├── Happy/
        ├── Angry/
        ├── Sad/
        └── Excited/
```

### 口スプライト（Mouthフォルダ）の構成

感情表現を使用するには、以下の構造で口スプライトを配置します：

```
mouth_dir/
└── Tomari/              # キャラクター名（任意）
    ├── Default/         # デフォルト表情
    │   ├── open.png     # 口を開けた状態
    │   ├── closed.png   # 口を閉じた状態
    │   ├── half.png     # 半開き
    │   ├── e.png        # 「え」の口（open/closed/halfとは異なる形状なら何でも可）
    │   └── u.png        # 「う」の口（open/closed/halfとは異なる形状なら何でも可）
    ├── Happy/           # 嬉しい
    │   ├── open.png
    │   ├── closed.png
    │   ├── half.png
    │   ├── e.png
    │   └── u.png
    ├── Angry/           # 怒り
    ├── Sad/             # 悲しみ
    └── Excited/         # 興奮
```

**重要事項**:
- 各フォルダに必ず `open.png`, `closed.png`, `half.png`, `e.png`, `u.png` の5ファイルを配置
- `e.png` と `u.png` は、`open.png` / `closed.png` / `half.png` とは**異なる形状**であれば何でも構いません（「え」「う」の口に限らず、任意の口の形状でOK）
- 画像形式: PNG（透過対応・RGBA）
- 推奨サイズ: 幅128px程度（アスペクト比 約1.5）
- 感情フォルダ名: `Default`, `Happy`, `Angry`, `Sad`, `Excited`（大文字小文字どちらでも可）

---

## 🎮 使い方

### GUI（推奨）

```bash
python mouth_track_gui.py
```

#### ワークフロー

1. **動画を選択**: 「選択…」ボタンで元動画（loop.mp4など）を選ぶ
2. **mouthフォルダを選択**: 口スプライトがあるフォルダを選ぶ
3. **キャラクターを選択**: 複数キャラがある場合はドロップダウンで選択
4. **① 解析→キャリブ**: クリックすると顔トラッキング→キャリブレーション画面へ
5. **② 口消し動画生成**: 口を消した動画を生成
6. **③ ライブ実行**: マイク入力でリアルタイム口パク開始

#### 設定項目

| 項目 | 説明 |
|------|------|
| pad（追跡余白） | トラッキング領域の大きさ（1.0〜3.0） |
| 口消し強さ | 口を消す範囲の大きさ（0.4〜0.9） |
| 影なじませ | ONで陰影を自然になじませる（顎の黒にじみが出る場合はOFF） |
| スムージング | 口の動きの滑らかさ（ゆっくり〜追従最優先） |
| 感情オート | 安定（配信向け）/ 標準 / キビキビ（ゲーム向け） |
| HUD | 現在の感情を画面に表示 |

---

## 🎯 キャリブレーション操作

「① 解析→キャリブ」実行後、キャリブレーション画面が表示されます。

### マウス操作

| 操作 | 機能 |
|------|------|
| 左ドラッグ | 口スプライトを移動 |
| ホイール | 拡大・縮小 |
| Ctrl + ホイール | 微調整 |
| 右ドラッグ | 回転 |

### キーボード操作

| キー | 機能 |
|------|------|
| 矢印キー | 微移動 |
| `+` / `-` | 拡大・縮小 |
| `z` / `x` | 回転（左回り / 右回り） |
| `[` / `]` | フレーム移動（プレビュー用） |
| `r` | リセット（初期値に戻す） |
| `Space` / `Enter` | 確定して保存 |
| `q` / `Esc` | キャンセル |

---

## 👀 口消し範囲プレビュー操作

「口消し範囲プレビュー」ボタンで、口消しの範囲を事前確認できます。

### 表示

- **赤色**: 実際に消去される領域（中心マスク）
- **黄色**: 陰影推定に使う外周領域（ring）
- **緑枠**: トラッキングされた口の四角形

### キーボード操作

| キー | 機能 |
|------|------|
| `Space` | 再生 / 一時停止 |
| `a` / `d` | 1フレーム戻る / 進む |
| `[` / `]` | 10フレーム戻る / 進む |
| `q` / `Esc` | 終了 |

---

## 🎁 おまけツール: 口消しチューナーGUI

```bash
python mouth_erase_tuner_gui.py
```

画像の口部分を削除できる単体ツールです。MotionPNGTuber本体とは独立して使用できます。

---

## ⌨️ コマンドライン使用

### 1. 顔トラッキング

#### 軽量版（YOLOv9・推奨）

```bash
python face_track_anime_detector.py \
    --video assets/assets01/loop.mp4 \
    --out assets/assets01/mouth_track.npz \
    --detector yolov9 \
    --device cpu \
    --pad 2.1
```

#### フル版（anime-face-detector）

```bash
python face_track_anime_detector.py \
    --video assets/assets01/loop.mp4 \
    --out assets/assets01/mouth_track.npz \
    --detector anime \
    --device auto \
    --pad 2.1
```

### 2. キャリブレーション

```bash
python calibrate_mouth_track.py \
    --video assets/assets01/loop.mp4 \
    --track assets/assets01/mouth_track.npz \
    --sprite mouth_dir/Tomari/Default/open.png \
    --out assets/assets01/mouth_track_calibrated.npz
```

### 3. 口消し動画生成

```bash
python auto_erase_mouth.py \
    --video assets/assets01/loop.mp4 \
    --track assets/assets01/mouth_track_calibrated.npz \
    --out assets/assets01/loop_mouthless.mp4 \
    --coverage 0.6
```

### 4. リアルタイム実行

```bash
python loop_lipsync_runtime_patched_emotion_auto.py \
    --loop-video assets/assets01/loop_mouthless.mp4 \
    --mouth-dir mouth_dir/Tomari \
    --track assets/assets01/mouth_track_calibrated.npz \
    --emotion-auto \
    --emotion-preset snappy
```

### デバイス指定

```bash
# GPU使用（自動選択）
--device auto

# 特定のGPU
--device cuda:0

# CPUのみ
--device cpu
```

---

## ❓ トラブルシューティング

### YOLOv9モデルが見つからない（軽量版）

初回実行時にモデルが自動ダウンロードされます。ネットワーク環境によっては手動ダウンロードが必要な場合があります：

```bash
# ダウンロード手順を表示
python yolov9_wholebody_detector.py --download
```

または以下から手動ダウンロード：
1. https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/459_YOLOv9-Wholebody25/resources_n.tar.gz
2. 展開して `models/yolov9_wholebody25/` に配置

### mmcv-full のインストールが失敗する（フル版）

> 💡 **ヒント**: mmcv-fullのインストールが難しい場合は、軽量版（YOLOv9）の使用を推奨します。

1. Visual C++ Build Tools がインストールされているか確認
2. Python のバージョンが 3.10 か確認
3. 以下を試す：
   ```bash
   pip cache purge
   mim install mmcv-full==1.7.0
   ```

### anime-face-detector が動かない（フル版）

```bash
pip uninstall anime-face-detector
pip install anime-face-detector==0.0.9 --no-cache-dir
```

### CUDA が認識されない

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
False の場合：
- NVIDIA ドライバが最新か確認
- CUDA Toolkit 11.8 がインストールされているか確認
- PyTorch を再インストール

### sounddevice でエラー

```bash
pip uninstall sounddevice
pip install sounddevice --no-cache-dir
```

オーディオデバイスが認識されない場合、Windowsのサウンド設定でマイクが有効か確認。

### 口の位置がズレる

1. 「キャリブのみ（やり直し）」でキャリブレーションをやり直す
2. `pad` の値を調整（大きくすると追跡領域が広がる）

### 口消しに黒いにじみが出る

「影なじませ」をOFFにする（GUIのチェックボックス）

---

## 🎭 感情判定について

音声から以下の5つの感情を自動判定します：

| 感情 | 判定基準 |
|------|----------|
| neutral | 標準状態 |
| happy | 高い声、明るいトーン |
| angry | 強い声、高エネルギー |
| sad | 低い声、静かなトーン |
| excited | 非常に高いエネルギー |

### プリセット

| プリセット | 特徴 |
|------------|------|
| 安定（配信向け） | 感情変化がゆっくり、チラつきにくい |
| 標準 | バランス重視 |
| キビキビ（ゲーム向け） | 感情変化が素早い、反応が良い |

---

## 📄 ライセンス

MIT License

---

## 🙏 謝辞

- [anime-face-detector](https://github.com/hysts/anime-face-detector)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) - YOLOv9-Wholebody25 モデル
