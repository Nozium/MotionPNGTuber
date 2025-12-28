#!/usr/bin/env python3
"""
export_onnx.py

anime-face-detector の YOLOv3 (顔検出) と HRNetV2 (ランドマーク検出) モデルを
ONNX 形式にエクスポートするスクリプト。

使用方法:
    # 1. 必要なパッケージをインストール
    pip install openmim
    mim install mmcv-full mmdet mmpose
    pip install anime-face-detector onnx onnxruntime

    # 2. スクリプト実行
    python scripts/export_onnx.py --output-dir models/onnx

出力:
    models/onnx/
    ├── anime_face_yolov3.onnx      # 顔検出モデル (~60MB)
    ├── anime_landmark_hrnetv2.onnx # ランドマーク検出モデル (~40MB)
    └── export_info.json            # エクスポート情報
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def check_dependencies() -> Tuple[bool, list]:
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import mmcv
    except ImportError:
        missing.append("mmcv-full (install via: mim install mmcv-full)")

    try:
        import mmdet
    except ImportError:
        missing.append("mmdet (install via: mim install mmdet)")

    try:
        import mmpose
    except ImportError:
        missing.append("mmpose (install via: mim install mmpose)")

    try:
        import anime_face_detector
    except ImportError:
        missing.append("anime-face-detector")

    return len(missing) == 0, missing


def get_model_paths() -> Dict[str, Path]:
    """Get paths to anime-face-detector model configs and checkpoints."""
    import anime_face_detector

    pkg_dir = Path(anime_face_detector.__file__).parent

    return {
        "yolov3_config": pkg_dir / "configs" / "mmdet" / "yolov3.py",
        "yolov3_checkpoint": "https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmdet_anime-face_yolov3.pth",
        "hrnetv2_config": pkg_dir / "configs" / "mmpose" / "hrnetv2.py",
        "hrnetv2_checkpoint": "https://github.com/hysts/anime-face-detector/releases/download/v0.0.1/mmpose_anime-face_hrnetv2.pth",
    }


def export_yolov3_onnx(
    config_path: Path,
    checkpoint_url: str,
    output_path: Path,
    input_size: Tuple[int, int] = (608, 608),
    opset_version: int = 11,
    simplify: bool = True,
) -> Dict[str, Any]:
    """Export YOLOv3 face detector to ONNX."""
    import torch
    from mmdet.apis import init_detector

    print(f"[YOLOv3] Loading model from {config_path}...")
    model = init_detector(str(config_path), checkpoint_url, device="cpu")
    model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])

    print(f"[YOLOv3] Exporting to ONNX (input shape: {list(dummy_input.shape)})...")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["boxes", "scores"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            import onnx
            print("[YOLOv3] Simplifying ONNX model...")
            model_onnx = onnx.load(str(output_path))
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, str(output_path))
        except ImportError:
            print("[YOLOv3] onnxsim not installed, skipping simplification")

    file_size = os.path.getsize(output_path)
    print(f"[YOLOv3] Exported: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    return {
        "model": "yolov3",
        "input_shape": [batch_size, 3, input_size[0], input_size[1]],
        "input_names": ["input"],
        "output_names": ["boxes", "scores"],
        "opset_version": opset_version,
        "file_size_bytes": file_size,
    }


def export_hrnetv2_onnx(
    config_path: Path,
    checkpoint_url: str,
    output_path: Path,
    input_size: Tuple[int, int] = (256, 256),
    opset_version: int = 11,
    simplify: bool = True,
) -> Dict[str, Any]:
    """Export HRNetV2 landmark detector to ONNX."""
    import torch
    from mmpose.apis import init_pose_model

    print(f"[HRNetV2] Loading model from {config_path}...")
    model = init_pose_model(str(config_path), checkpoint_url, device="cpu")
    model.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])

    print(f"[HRNetV2] Exporting to ONNX (input shape: {list(dummy_input.shape)})...")

    # Get the backbone + head for export
    # mmpose models have complex forward, we need to trace only inference path
    class PoseModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.backbone
            self.keypoint_head = model.keypoint_head

        def forward(self, x):
            features = self.backbone(x)
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Take last feature map
            heatmaps = self.keypoint_head(features)
            return heatmaps

    wrapped_model = PoseModelWrapper(model)
    wrapped_model.eval()

    # Export
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "heatmaps": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            import onnx
            print("[HRNetV2] Simplifying ONNX model...")
            model_onnx = onnx.load(str(output_path))
            model_onnx, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_onnx, str(output_path))
        except ImportError:
            print("[HRNetV2] onnxsim not installed, skipping simplification")

    file_size = os.path.getsize(output_path)
    print(f"[HRNetV2] Exported: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    return {
        "model": "hrnetv2",
        "input_shape": [batch_size, 3, input_size[0], input_size[1]],
        "input_names": ["input"],
        "output_names": ["heatmaps"],
        "num_keypoints": 28,
        "heatmap_size": [64, 64],
        "opset_version": opset_version,
        "file_size_bytes": file_size,
    }


def verify_onnx_model(onnx_path: Path) -> bool:
    """Verify ONNX model is valid."""
    import onnx

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"[Verify] {onnx_path.name}: OK")
        return True
    except Exception as e:
        print(f"[Verify] {onnx_path.name}: FAILED - {e}")
        return False


def test_onnx_inference(onnx_path: Path, input_shape: list) -> bool:
    """Test ONNX model inference with onnxruntime."""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        outputs = sess.run(None, {"input": dummy_input})
        print(f"[Test] {onnx_path.name}: OK (output shapes: {[o.shape for o in outputs]})")
        return True
    except Exception as e:
        print(f"[Test] {onnx_path.name}: FAILED - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export anime-face-detector models to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--yolov3-input-size",
        type=int,
        nargs=2,
        default=[608, 608],
        help="YOLOv3 input size (H W)",
    )
    parser.add_argument(
        "--hrnetv2-input-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="HRNetV2 input size (H W)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX verification",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip inference test",
    )
    args = parser.parse_args()

    # Check dependencies
    print("=" * 60)
    print("Checking dependencies...")
    ok, missing = check_dependencies()
    if not ok:
        print("\nMissing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies and try again.")
        print("\nInstallation commands:")
        print("  pip install openmim")
        print("  mim install mmcv-full mmdet mmpose")
        print("  pip install anime-face-detector onnx onnxruntime")
        sys.exit(1)
    print("All dependencies OK!")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model paths
    print("\n" + "=" * 60)
    print("Getting model paths...")
    paths = get_model_paths()

    export_info = {
        "exported_at": datetime.now().isoformat(),
        "opset_version": args.opset_version,
        "models": {},
    }

    # Export YOLOv3
    print("\n" + "=" * 60)
    print("Exporting YOLOv3 (face detection)...")
    yolov3_path = output_dir / "anime_face_yolov3.onnx"
    try:
        info = export_yolov3_onnx(
            paths["yolov3_config"],
            paths["yolov3_checkpoint"],
            yolov3_path,
            input_size=tuple(args.yolov3_input_size),
            opset_version=args.opset_version,
            simplify=not args.no_simplify,
        )
        export_info["models"]["yolov3"] = info
    except Exception as e:
        print(f"[ERROR] YOLOv3 export failed: {e}")
        import traceback
        traceback.print_exc()

    # Export HRNetV2
    print("\n" + "=" * 60)
    print("Exporting HRNetV2 (landmark detection)...")
    hrnetv2_path = output_dir / "anime_landmark_hrnetv2.onnx"
    try:
        info = export_hrnetv2_onnx(
            paths["hrnetv2_config"],
            paths["hrnetv2_checkpoint"],
            hrnetv2_path,
            input_size=tuple(args.hrnetv2_input_size),
            opset_version=args.opset_version,
            simplify=not args.no_simplify,
        )
        export_info["models"]["hrnetv2"] = info
    except Exception as e:
        print(f"[ERROR] HRNetV2 export failed: {e}")
        import traceback
        traceback.print_exc()

    # Verify models
    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("Verifying ONNX models...")
        for path in [yolov3_path, hrnetv2_path]:
            if path.exists():
                verify_onnx_model(path)

    # Test inference
    if not args.skip_test:
        print("\n" + "=" * 60)
        print("Testing inference...")
        for model_name, path in [("yolov3", yolov3_path), ("hrnetv2", hrnetv2_path)]:
            if path.exists() and model_name in export_info["models"]:
                input_shape = export_info["models"][model_name]["input_shape"]
                test_onnx_inference(path, input_shape)

    # Save export info
    info_path = output_dir / "export_info.json"
    with open(info_path, "w") as f:
        json.dump(export_info, f, indent=2)
    print(f"\nExport info saved to: {info_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    total_size = 0
    for model_name, info in export_info.get("models", {}).items():
        size_mb = info.get("file_size_bytes", 0) / 1024 / 1024
        total_size += size_mb
        print(f"  {model_name}: {size_mb:.1f} MB")
    print(f"  Total: {total_size:.1f} MB")

    if total_size > 45:
        print(f"\n[WARNING] Total size ({total_size:.1f} MB) exceeds 45 MB limit!")
    else:
        print(f"\n[OK] Total size ({total_size:.1f} MB) is within 45 MB limit.")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
