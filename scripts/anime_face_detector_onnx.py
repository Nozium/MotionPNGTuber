#!/usr/bin/env python3
"""
anime_face_detector_onnx.py

ONNX Runtime を使用したアニメ顔検出器。
mmcv/mmdet/mmpose に依存せず、軽量な環境で動作可能。

使用方法:
    from anime_face_detector_onnx import AnimeDetectorONNX

    detector = AnimeDetectorONNX(
        yolov3_path="models/onnx/anime_face_yolov3.onnx",
        hrnetv2_path="models/onnx/anime_landmark_hrnetv2.onnx"
    )

    # BGR画像で呼び出し
    results = detector(image_bgr)
    # results: [{"bbox": [x1,y1,x2,y2,conf], "keypoints": (28,3)}, ...]

依存パッケージ:
    pip install onnxruntime numpy opencv-python
    # GPU使用時: pip install onnxruntime-gpu
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2


class AnimeDetectorONNX:
    """
    ONNX Runtime ベースのアニメ顔検出器。

    anime-face-detector と同じインターフェースで、
    bbox と 28点キーポイントを出力。
    """

    # ImageNet normalization parameters
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        yolov3_path: Union[str, Path],
        hrnetv2_path: Union[str, Path],
        device: str = "cpu",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        box_scale_factor: float = 1.1,
    ):
        """
        Args:
            yolov3_path: YOLOv3 ONNX モデルパス
            hrnetv2_path: HRNetV2 ONNX モデルパス
            device: "cpu" or "cuda" (cuda使用時は onnxruntime-gpu が必要)
            score_threshold: 顔検出の信頼度閾値
            nms_threshold: NMS の IoU 閾値
            box_scale_factor: ランドマーク検出用に bbox を拡大する係数
        """
        import onnxruntime as ort

        # Select providers based on device
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Load models
        self.face_detector = ort.InferenceSession(str(yolov3_path), providers=providers)
        self.landmark_detector = ort.InferenceSession(str(hrnetv2_path), providers=providers)

        # Get input shapes
        face_input = self.face_detector.get_inputs()[0]
        landmark_input = self.landmark_detector.get_inputs()[0]

        self.face_input_name = face_input.name
        self.face_input_size = (face_input.shape[2], face_input.shape[3])  # (H, W)

        self.landmark_input_name = landmark_input.name
        self.landmark_input_size = (landmark_input.shape[2], landmark_input.shape[3])  # (H, W)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.box_scale_factor = box_scale_factor

        self._device = device

    def __call__(
        self,
        image: np.ndarray,
        boxes: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        画像から顔を検出し、各顔のランドマークを推定。

        Args:
            image: BGR 画像 (H, W, 3)
            boxes: Optional - 顔 bbox のリスト。指定時は顔検出をスキップ

        Returns:
            検出結果のリスト。各要素は:
            {
                "bbox": np.ndarray [x1, y1, x2, y2, confidence],
                "keypoints": np.ndarray (28, 3) - [x, y, confidence]
            }
        """
        if boxes is None:
            boxes = self._detect_faces(image)

        if len(boxes) == 0:
            return []

        results = []
        for box in boxes:
            # Scale box for landmark detection
            scaled_box = self._scale_box(box, image.shape[:2])

            # Detect landmarks
            keypoints = self._detect_landmarks(image, scaled_box)

            results.append({
                "bbox": box,
                "keypoints": keypoints,
            })

        return results

    def _preprocess(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        画像を前処理してモデル入力形式に変換。

        Returns:
            (preprocessed_image, scale, padding)
        """
        h, w = image.shape[:2]
        th, tw = target_size

        # Calculate scale to fit
        scale = min(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = (tw - new_w) // 2
        pad_h = (th - new_h) // 2

        padded = np.zeros((th, tw, 3), dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD

        # HWC -> NCHW
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        return tensor.astype(np.float32), scale, (pad_w, pad_h)

    def _detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """YOLOv3 で顔を検出。"""
        h, w = image.shape[:2]

        # Preprocess
        tensor, scale, (pad_w, pad_h) = self._preprocess(image, self.face_input_size)

        # Inference
        outputs = self.face_detector.run(None, {self.face_input_name: tensor})

        # Parse outputs - format depends on export configuration
        # Typically: boxes (N, 4) and scores (N,) or combined (N, 5)
        boxes = self._parse_yolov3_output(outputs, scale, pad_w, pad_h, w, h)

        # NMS
        if len(boxes) > 0:
            boxes = self._nms(boxes)

        return boxes

    def _parse_yolov3_output(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> List[np.ndarray]:
        """YOLOv3 出力をパース。"""
        # Output format varies by export method
        # Try common formats
        boxes = []

        if len(outputs) >= 2:
            # Separate boxes and scores
            raw_boxes = outputs[0]  # (batch, num_boxes, 4) or (num_boxes, 4)
            raw_scores = outputs[1]  # (batch, num_boxes, num_classes) or similar

            if raw_boxes.ndim == 3:
                raw_boxes = raw_boxes[0]
            if raw_scores.ndim == 3:
                raw_scores = raw_scores[0]
            if raw_scores.ndim == 2:
                raw_scores = raw_scores.max(axis=1)

            for i, (box, score) in enumerate(zip(raw_boxes, raw_scores)):
                if score >= self.score_threshold:
                    x1, y1, x2, y2 = box[:4]
                    # Remove padding and scale back
                    x1 = (x1 - pad_w) / scale
                    y1 = (y1 - pad_h) / scale
                    x2 = (x2 - pad_w) / scale
                    y2 = (y2 - pad_h) / scale
                    # Clip to image bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    boxes.append(np.array([x1, y1, x2, y2, float(score)], dtype=np.float32))

        elif len(outputs) == 1:
            # Combined format (num_boxes, 5+)
            raw = outputs[0]
            if raw.ndim == 3:
                raw = raw[0]

            for det in raw:
                score = det[4] if len(det) >= 5 else det[-1]
                if score >= self.score_threshold:
                    x1, y1, x2, y2 = det[:4]
                    x1 = (x1 - pad_w) / scale
                    y1 = (y1 - pad_h) / scale
                    x2 = (x2 - pad_w) / scale
                    y2 = (y2 - pad_h) / scale
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    boxes.append(np.array([x1, y1, x2, y2, float(score)], dtype=np.float32))

        return boxes

    def _nms(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        scores = boxes_arr[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def _scale_box(
        self,
        box: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """bbox を拡大（ランドマーク検出用）。"""
        x1, y1, x2, y2 = box[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * self.box_scale_factor
        h = (y2 - y1) * self.box_scale_factor

        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(img_shape[1], cx + w / 2)
        y2 = min(img_shape[0], cy + h / 2)

        return np.array([x1, y1, x2, y2, box[4]], dtype=np.float32)

    def _detect_landmarks(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        """HRNetV2 でランドマークを検出。"""
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop face region
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((28, 3), dtype=np.float32)

        crop_h, crop_w = crop.shape[:2]

        # Preprocess (no padding, just resize)
        th, tw = self.landmark_input_size
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # Inference
        outputs = self.landmark_detector.run(None, {self.landmark_input_name: tensor})
        heatmaps = outputs[0]  # (1, 28, H, W)

        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]  # (28, H, W)

        # Decode heatmaps to keypoints
        keypoints = self._decode_heatmaps(heatmaps, crop_w, crop_h)

        # Transform back to original image coordinates
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1

        return keypoints

    def _decode_heatmaps(
        self,
        heatmaps: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """
        ヒートマップからキーポイント座標を抽出。

        Args:
            heatmaps: (28, H, W) ヒートマップ
            orig_w, orig_h: 元のクロップサイズ

        Returns:
            keypoints: (28, 3) - [x, y, confidence]
        """
        num_keypoints, hm_h, hm_w = heatmaps.shape
        keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

        for i in range(num_keypoints):
            hm = heatmaps[i]

            # Find max location
            max_val = hm.max()
            if max_val > 0:
                max_idx = hm.argmax()
                max_y = max_idx // hm_w
                max_x = max_idx % hm_w

                # Sub-pixel refinement using Taylor expansion
                if 0 < max_x < hm_w - 1 and 0 < max_y < hm_h - 1:
                    dx = (hm[max_y, max_x + 1] - hm[max_y, max_x - 1]) / 2
                    dy = (hm[max_y + 1, max_x] - hm[max_y - 1, max_x]) / 2
                    max_x = max_x + np.clip(dx, -0.5, 0.5)
                    max_y = max_y + np.clip(dy, -0.5, 0.5)

                # Scale to original size
                x = (max_x + 0.5) * orig_w / hm_w
                y = (max_y + 0.5) * orig_h / hm_h

                keypoints[i] = [x, y, max_val]

        return keypoints


def create_detector(
    model_name: str = "yolov3",
    device: str = "cpu",
    onnx_dir: Union[str, Path] = "models/onnx",
) -> AnimeDetectorONNX:
    """
    anime-face-detector 互換の create_detector 関数。

    Args:
        model_name: "yolov3" (現在これのみサポート)
        device: "cpu" or "cuda:0" etc.
        onnx_dir: ONNX モデルが保存されているディレクトリ

    Returns:
        AnimeDetectorONNX インスタンス
    """
    onnx_dir = Path(onnx_dir)

    yolov3_path = onnx_dir / "anime_face_yolov3.onnx"
    hrnetv2_path = onnx_dir / "anime_landmark_hrnetv2.onnx"

    if not yolov3_path.exists():
        raise FileNotFoundError(f"YOLOv3 model not found: {yolov3_path}")
    if not hrnetv2_path.exists():
        raise FileNotFoundError(f"HRNetV2 model not found: {hrnetv2_path}")

    return AnimeDetectorONNX(
        yolov3_path=yolov3_path,
        hrnetv2_path=hrnetv2_path,
        device=device,
    )


# Test function
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--onnx-dir", default="models/onnx", help="ONNX models directory")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Failed to load image: {args.image}")
        return 1

    # Create detector
    detector = create_detector(device=args.device, onnx_dir=args.onnx_dir)

    # Detect
    results = detector(image)

    print(f"Detected {len(results)} faces")

    # Draw results
    for r in results:
        bbox = r["bbox"]
        keypoints = r["keypoints"]

        # Draw bbox
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            image, f"{conf:.2f}", (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # Draw keypoints
        for i, (x, y, c) in enumerate(keypoints):
            if c > 0.3:
                color = (0, 255, 255) if i in [24, 25, 26, 27] else (255, 0, 255)
                cv2.circle(image, (int(x), int(y)), 2, color, -1)

    cv2.imwrite(args.output, image)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
