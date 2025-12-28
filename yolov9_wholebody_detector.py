#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolov9_wholebody_detector.py

YOLOv9-Wholebody25 ONNX モデルを使用した軽量顔・口検出器。
anime-face-detector (mmdet/mmpose/mmcv) の代替として使用可能。

依存関係:
    pip install onnxruntime numpy opencv-python requests

使い方:
    from yolov9_wholebody_detector import YOLOv9WholebodyDetector

    detector = YOLOv9WholebodyDetector()
    results = detector(frame)  # BGR image
    # results = [{'face_bbox': [...], 'mouth_bbox': [...], 'head_direction': 'Front', ...}, ...]
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("[warn] onnxruntime not installed. Run: pip install onnxruntime")


# モデルダウンロード設定
MODEL_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/459_YOLOv9-Wholebody25/resources_n.tar.gz"
MODEL_DIR = Path(__file__).parent / "models" / "yolov9_wholebody25"
MODEL_FILENAME = "yolov9_e_wholebody25_post_0100_1x3x480x640.onnx"


# クラスID定義 (YOLOv9-Wholebody25)
CLASS_NAMES = [
    "Body",           # 0
    "Adult",          # 1
    "Child",          # 2
    "Male",           # 3
    "Female",         # 4
    "Body_with_Wheelchair",  # 5
    "Body_with_Crutches",    # 6
    "Head",           # 7
    "Front",          # 8
    "Right_Front",    # 9
    "Right_Side",     # 10
    "Right_Back",     # 11
    "Back",           # 12
    "Left_Back",      # 13
    "Left_Side",      # 14
    "Left_Front",     # 15
    "Face",           # 16
    "Eye",            # 17
    "Nose",           # 18
    "Mouth",          # 19
    "Ear",            # 20
    "Hand",           # 21
    "Hand_Left",      # 22
    "Hand_Right",     # 23
    "Foot",           # 24
]

# クラスIDマッピング
CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 頭部方向 → 角度マッピング
HEAD_DIRECTION_TO_ANGLE = {
    "Front": 0.0,
    "Right_Front": -22.5,
    "Right_Side": -45.0,
    "Right_Back": -67.5,
    "Back": 0.0,  # 後ろ向きは0度として扱う
    "Left_Back": 67.5,
    "Left_Side": 45.0,
    "Left_Front": 22.5,
}

# 頭部方向クラスID
HEAD_DIRECTION_IDS = {8, 9, 10, 11, 12, 13, 14, 15}


@dataclass
class Detection:
    """検出結果を格納するデータクラス"""
    classid: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.classid] if 0 <= self.classid < len(CLASS_NAMES) else "Unknown"

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def to_bbox(self) -> np.ndarray:
        """[x1, y1, x2, y2, conf] 形式で返す"""
        return np.array([self.x1, self.y1, self.x2, self.y2, self.score], dtype=np.float32)


def download_model(url: str = MODEL_URL, model_dir: Path = MODEL_DIR) -> Path:
    """
    モデルをダウンロードして展開する。
    すでにモデルファイルが存在する場合はスキップ。
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME

    if model_path.exists():
        print(f"[info] Model already exists: {model_path}")
        return model_path

    print(f"[info] Downloading model from {url}...")

    # ダウンロード
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        print(f"[info] Downloaded to {tmp_path}")

        # 展開
        print(f"[info] Extracting to {model_dir}...")
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(model_dir)

        # モデルファイルを探す
        if model_path.exists():
            print(f"[info] Model ready: {model_path}")
            return model_path

        # 別のファイル名で探す
        onnx_files = list(model_dir.rglob("*.onnx"))
        if onnx_files:
            # 480x640のモデルを優先
            for f in onnx_files:
                if "480x640" in f.name:
                    print(f"[info] Found model: {f}")
                    return f
            # 見つからなければ最初のONNXファイル
            print(f"[info] Found model: {onnx_files[0]}")
            return onnx_files[0]

        raise FileNotFoundError(f"No ONNX model found in {model_dir}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class YOLOv9WholebodyDetector:
    """
    YOLOv9-Wholebody25 ONNX モデルを使用した検出器。

    出力:
        - Face検出 (BBox)
        - Mouth検出 (BBox)
        - Head方向 (8方向)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        score_threshold: float = 0.35,
        auto_download: bool = True,
    ):
        """
        Args:
            model_path: ONNXモデルのパス。Noneの場合は自動ダウンロード。
            device: "cpu", "cuda", "cuda:0" など
            score_threshold: 検出スコアの閾値
            auto_download: モデルが見つからない場合に自動ダウンロード
        """
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required. Run: pip install onnxruntime")

        self.score_threshold = score_threshold

        # モデルパスの解決
        if model_path is None:
            if auto_download:
                model_path = str(download_model())
            else:
                model_path = str(MODEL_DIR / MODEL_FILENAME)

        if not os.path.exists(model_path):
            if auto_download:
                model_path = str(download_model())
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        # プロバイダー設定
        providers = self._get_providers(device)
        print(f"[info] Loading model: {model_path}")
        print(f"[info] Providers: {providers}")

        # セッション作成
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # ERROR only

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        # 入出力情報を取得
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]
        self.output_name = self.session.get_outputs()[0].name

        # 入力サイズ（動的な場合はデフォルト値を使用）
        self.input_height = self.input_shape[2] if isinstance(self.input_shape[2], int) else 480
        self.input_width = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

        print(f"[info] Input shape: {self.input_shape} ({self.input_height}x{self.input_width})")

    def _get_providers(self, device: str) -> List[str]:
        """デバイス指定からONNX Runtimeプロバイダーを取得"""
        available = ort.get_available_providers()

        if device.startswith("cuda"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print(f"[warn] CUDA not available, falling back to CPU")
                return ["CPUExecutionProvider"]
        elif device == "tensorrt":
            if "TensorrtExecutionProvider" in available:
                return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print(f"[warn] TensorRT not available, falling back to CUDA/CPU")
                return self._get_providers("cuda")
        else:
            return ["CPUExecutionProvider"]

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        前処理: リサイズ、正規化、チャネル順変換

        Returns:
            processed: (1, 3, H, W) float32 テンソル
            scale_x: 元画像への変換スケール (x方向)
            scale_y: 元画像への変換スケール (y方向)
        """
        orig_h, orig_w = image.shape[:2]

        # リサイズ
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # BGR -> RGB (モデルがRGB入力の場合)
        # Note: このモデルはBGR入力のようなので変換しない

        # HWC -> CHW
        transposed = resized.transpose(2, 0, 1)

        # float32に変換（正規化はモデルに組み込まれている）
        processed = np.ascontiguousarray(transposed, dtype=np.float32)

        # バッチ次元追加
        processed = np.expand_dims(processed, axis=0)

        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        return processed, scale_x, scale_y

    def _postprocess(
        self,
        outputs: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> List[Detection]:
        """
        後処理: 出力テンソルをDetectionリストに変換

        outputs: [N, 7] - [batch_id, class_id, score, x1, y1, x2, y2]
        """
        detections = []

        for row in outputs:
            batch_id, class_id, score, x1, y1, x2, y2 = row

            if score < self.score_threshold:
                continue

            # 元画像サイズにスケール変換
            det = Detection(
                classid=int(class_id),
                score=float(score),
                x1=float(x1) * scale_x,
                y1=float(y1) * scale_y,
                x2=float(x2) * scale_x,
                y2=float(y2) * scale_y,
            )
            detections.append(det)

        return detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        画像から検出を実行

        Args:
            image: BGR画像 (H, W, 3) uint8

        Returns:
            検出結果のリスト
        """
        # 前処理
        processed, scale_x, scale_y = self._preprocess(image)

        # 推論
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: processed},
        )[0]

        # 後処理
        detections = self._postprocess(outputs, scale_x, scale_y)

        return detections

    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        anime-face-detector互換のインターフェース

        Args:
            image: BGR画像 (H, W, 3) uint8

        Returns:
            検出結果リスト。各要素は:
            {
                'face_bbox': [x1, y1, x2, y2, conf] or None,
                'mouth_bbox': [x1, y1, x2, y2, conf] or None,
                'head_direction': str or None,  # 'Front', 'Right_Side', etc.
                'head_angle': float,  # 角度（度）
                'all_detections': List[Detection],  # 全検出結果
            }
        """
        all_detections = self.detect(image)

        # Face検出を探す
        faces = [d for d in all_detections if d.classid == CLASS_ID["Face"]]

        if not faces:
            return []

        results = []

        for face in faces:
            face_bbox = face.to_bbox()
            face_center = face.center

            # この顔に対応するMouth検出を探す（顔bbox内）
            mouth_bbox = None
            for d in all_detections:
                if d.classid == CLASS_ID["Mouth"]:
                    # Mouthの中心がFace bbox内にあるか確認
                    mx, my = d.center
                    if face.x1 <= mx <= face.x2 and face.y1 <= my <= face.y2:
                        if mouth_bbox is None or d.score > mouth_bbox[4]:
                            mouth_bbox = d.to_bbox()

            # 頭部方向を探す（Head bbox内）
            head_direction = None
            head_angle = 0.0

            # まずHeadを探す
            heads = [d for d in all_detections if d.classid == CLASS_ID["Head"]]
            for head in heads:
                # HeadがFaceと重なるか確認
                hcx, hcy = head.center
                if face.x1 - 50 <= hcx <= face.x2 + 50 and face.y1 - 50 <= hcy <= face.y2 + 50:
                    # このHeadに対応する方向検出を探す
                    for d in all_detections:
                        if d.classid in HEAD_DIRECTION_IDS:
                            dx, dy = d.center
                            if head.x1 <= dx <= head.x2 and head.y1 <= dy <= head.y2:
                                if head_direction is None or d.score > 0.5:
                                    head_direction = d.class_name
                                    head_angle = HEAD_DIRECTION_TO_ANGLE.get(head_direction, 0.0)
                    break

            results.append({
                'face_bbox': face_bbox,
                'mouth_bbox': mouth_bbox,
                'head_direction': head_direction,
                'head_angle': head_angle,
                'all_detections': all_detections,
            })

        return results


def mouth_bbox_to_quad(
    mouth_bbox: np.ndarray,
    head_angle: float = 0.0,
    pad: float = 1.2,
    sprite_aspect: float = 1.0,
) -> np.ndarray:
    """
    口のBBoxをQuadに変換（回転対応）

    Args:
        mouth_bbox: [x1, y1, x2, y2, conf]
        head_angle: 頭部の傾き角度（度）
        pad: パディング係数
        sprite_aspect: スプライトのアスペクト比 (w/h)

    Returns:
        quad: (4, 2) - [TL, TR, BR, BL]
    """
    x1, y1, x2, y2 = mouth_bbox[:4]

    # 中心とサイズ
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * pad
    h = (y2 - y1) * pad

    # アスペクト比調整
    if sprite_aspect > 0:
        h_from_aspect = w / sprite_aspect
        h = max(h, h_from_aspect)

    hw, hh = w / 2, h / 2

    # ローカル座標（回転前）
    quad_local = np.array([
        [-hw, -hh],  # TL
        [+hw, -hh],  # TR
        [+hw, +hh],  # BR
        [-hw, +hh],  # BL
    ], dtype=np.float32)

    # 回転
    angle_rad = np.radians(head_angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a],
    ], dtype=np.float32)

    quad_rotated = quad_local @ R.T

    # 中心に移動
    quad = quad_rotated + np.array([cx, cy], dtype=np.float32)

    return quad.astype(np.float32)


def print_download_instructions():
    """手動ダウンロード手順を表示"""
    print("""
=== YOLOv9-Wholebody25 モデルのダウンロード手順 ===

1. 以下のURLからモデルをダウンロード:
   https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/459_YOLOv9-Wholebody25/resources_n.tar.gz

2. ダウンロードしたファイルを展開:
   tar -xzf resources_n.tar.gz

3. ONNXファイルを以下のディレクトリに配置:
   {model_dir}/

4. 推奨モデルファイル:
   - yolov9_n_wholebody25_post_0100_1x3x480x640.onnx (480x640入力、最軽量)
   - yolov9_n_wholebody25_post_0100_1x3x256x320.onnx (320x256入力、より高速)

または、GitHubからダウンロードスクリプトを使用:
   cd {model_dir}
   curl -O https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/main/459_YOLOv9-Wholebody25/download_n.sh
   bash download_n.sh
""".format(model_dir=MODEL_DIR))


def test_without_model():
    """モデルなしでロジックをテスト"""
    print("=== YOLOv9-Wholebody25 Logic Test ===")

    # mouth_bbox_to_quad テスト
    print("[test] mouth_bbox_to_quad...")
    test_bbox = np.array([100, 200, 150, 230, 0.9], dtype=np.float32)

    quad_0 = mouth_bbox_to_quad(test_bbox, head_angle=0.0, pad=1.5)
    print(f"  Input: {test_bbox[:4]}")
    print(f"  Output (0 deg): {quad_0[0]} ...")

    quad_15 = mouth_bbox_to_quad(test_bbox, head_angle=15.0, pad=1.5)
    print(f"  Output (15 deg): {quad_15[0]} ...")

    # Detection テスト
    print("[test] Detection class...")
    det = Detection(classid=16, score=0.85, x1=100, y1=200, x2=150, y2=230)
    print(f"  {det.class_name}: center={det.center}, area={det.area}")

    print("\n[OK] All logic tests passed!")


def demo():
    """デモ: Webカメラで動作確認"""
    print("=== YOLOv9-Wholebody25 Detector Demo ===")

    try:
        detector = YOLOv9WholebodyDetector(device="cpu")
    except FileNotFoundError:
        print("[error] Model not found!")
        print_download_instructions()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] Cannot open webcam")
        return

    print("[info] Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 検出
        results = detector(frame)

        # 描画
        for res in results:
            # Face
            if res['face_bbox'] is not None:
                x1, y1, x2, y2, conf = res['face_bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {conf:.2f}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Mouth
            if res['mouth_bbox'] is not None:
                x1, y1, x2, y2, conf = res['mouth_bbox']
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Mouth {conf:.2f}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Quad描画
                quad = mouth_bbox_to_quad(res['mouth_bbox'], res['head_angle'])
                pts = quad.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Head direction
            if res['head_direction']:
                cv2.putText(frame, f"Head: {res['head_direction']} ({res['head_angle']:.1f}deg)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("YOLOv9-Wholebody25 Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv9-Wholebody25 Detector")
    parser.add_argument("--demo", action="store_true", help="Run webcam demo")
    parser.add_argument("--test", action="store_true", help="Run logic tests (no model needed)")
    parser.add_argument("--download", action="store_true", help="Show download instructions")
    args = parser.parse_args()

    if args.download:
        print_download_instructions()
    elif args.test:
        test_without_model()
    elif args.demo:
        demo()
    else:
        print("Usage:")
        print("  python yolov9_wholebody_detector.py --demo     # Webcam demo")
        print("  python yolov9_wholebody_detector.py --test     # Logic test")
        print("  python yolov9_wholebody_detector.py --download # Download instructions")


if __name__ == "__main__":
    main()
