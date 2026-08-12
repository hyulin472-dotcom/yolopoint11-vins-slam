#!/usr/bin/env python3
"""Run Fast-FoundationStereo on KITTI pairs and save disparity/depth visualizations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

from common import load_config, resolve_path


def read_projection_matrices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, data = line.split(":", 1)
        numbers = np.fromstring(data, sep=" ", dtype=np.float64)
        if numbers.size == 12:
            values[name] = numbers.reshape(3, 4)
    if "P2" not in values or "P3" not in values:
        raise ValueError(f"KITTI calibration must contain P2 and P3: {path}")
    return values["P2"], values["P3"]


def camera_center_x(projection: np.ndarray) -> float:
    intrinsic = projection[:, :3]
    translation = np.linalg.solve(intrinsic, projection[:, 3])
    return float(-translation[0])


def providers(name: str, cache_path: Path):
    available = ort.get_available_providers()
    if name == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(f"TensorRT provider unavailable: {available}")
        options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_path),
        }
        return [("TensorrtExecutionProvider", options), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if name == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDA provider unavailable: {available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preprocess(image_bgr: np.ndarray, width: int, height: int, component) -> np.ndarray:
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    chw = rgb.transpose(2, 0, 1)[None]
    mode = component.get("normalization", "raw_0_255")
    if mode == "imagenet_0_255":
        mean = np.asarray(component["normalization_mean"], dtype=np.float32)[None, :, None, None]
        std = np.asarray(component["normalization_std"], dtype=np.float32)[None, :, None, None]
        chw = (chw - mean) / std
    elif mode != "raw_0_255":
        raise ValueError(f"unsupported normalization: {mode}")
    return np.ascontiguousarray(chw)


def photometric_error(left_bgr: np.ndarray, right_bgr: np.ndarray, disparity: np.ndarray) -> float:
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    yy, xx = np.indices(disparity.shape, dtype=np.float32)
    right_x = xx - disparity
    warped = cv2.remap(right, right_x, yy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid = (right_x >= 0) & (disparity > 0) & np.isfinite(warped)
    if not np.any(valid):
        return float("nan")
    error = np.abs(left - warped)[valid]
    return float(np.mean(np.minimum(error, 0.2)))


def save_panel(path: Path, frame: str, left_bgr: np.ndarray, right_bgr: np.ndarray,
               disparity: np.ndarray, depth: np.ndarray, max_depth: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"KITTI {frame} - original left")
    axes[0, 1].imshow(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("original right")
    valid_disparity = disparity[disparity > 0]
    vmax = float(np.percentile(valid_disparity, 99)) if valid_disparity.size else 1.0
    im0 = axes[1, 0].imshow(disparity, cmap="turbo", vmin=0, vmax=max(vmax, 1.0))
    axes[1, 0].set_title("Fast-FoundationStereo disparity (model pixels)")
    fig.colorbar(im0, ax=axes[1, 0], label="px", fraction=0.025)
    masked = np.ma.masked_invalid(depth)
    im1 = axes[1, 1].imshow(masked, cmap="turbo_r", vmin=0, vmax=max_depth)
    axes[1, 1].set_title(f"metric depth (0-{max_depth:g} m)")
    fig.colorbar(im1, ax=axes[1, 1], label="m", fraction=0.025)
    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def contact_tile(left_bgr: np.ndarray, depth: np.ndarray, frame: str, max_depth: float) -> np.ndarray:
    normalized = np.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    normalized = np.clip(normalized / max_depth, 0, 1)
    depth_color = cv2.applyColorMap(np.uint8((1.0 - normalized) * 255), cv2.COLORMAP_TURBO)
    left = cv2.resize(left_bgr, (620, 188), interpolation=cv2.INTER_AREA)
    depth_color = cv2.resize(depth_color, (620, 188), interpolation=cv2.INTER_NEAREST)
    tile = np.vstack((left, depth_color))
    cv2.putText(tile, f"frame {frame}: RGB / metric depth", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, help="KITTI sequence directory containing image_2/image_3/calib.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--provider", choices=("tensorrt", "cuda", "cpu"), default="cuda")
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    args = parser.parse_args()
    if args.count < 10:
        raise ValueError("count must be at least 10 for this visualization test")

    config = load_config(args.config)
    component = config["components"]["stereo"]
    model = resolve_path(config, component["onnx_path"], required=True)
    cache = resolve_path(config, component["tensorrt"]["ort_cache_path"])
    dataset = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    (output / "arrays").mkdir(exist_ok=True)

    p2, p3 = read_projection_matrices(dataset / "calib.txt")
    focal = float(p2[0, 0])
    baseline = abs(camera_center_x(p3) - camera_center_x(p2))
    height, width = map(int, component["input_size"])
    session = ort.InferenceSession(str(model), providers=providers(args.provider, cache))
    if args.provider == "tensorrt" and "TensorrtExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"TensorRT silently fell back: {session.get_providers()}")

    rows, tiles = [], []
    for index in range(args.start, args.start + args.count * args.stride, args.stride):
        frame = f"{index:06d}"
        left_path = dataset / "image_2" / f"{frame}.png"
        right_path = dataset / "image_3" / f"{frame}.png"
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError(f"missing stereo pair: {left_path}, {right_path}")
        left_tensor = preprocess(left, width, height, component)
        right_tensor = preprocess(right, width, height, component)
        start_time = time.perf_counter()
        disparity = session.run(None, {"left_image": left_tensor, "right_image": right_tensor})[0][0, 0]
        inference_ms = (time.perf_counter() - start_time) * 1000.0

        # Convert model-pixel disparity to original-pixel disparity before Z=fB/d.
        original_scale = left.shape[1] / float(width)
        original_disparity = disparity * original_scale
        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = original_disparity > 1e-6
        depth[valid] = focal * baseline / original_disparity[valid]
        display_valid = valid & (depth >= args.min_depth) & (depth <= args.max_depth)
        display_depth = np.where(display_valid, depth, np.nan)

        np.save(output / "arrays" / f"{frame}_disparity.npy", disparity)
        np.save(output / "arrays" / f"{frame}_depth_m.npy", depth)
        depth_mm = np.zeros_like(depth, dtype=np.uint16)
        depth_mm[display_valid] = np.clip(depth[display_valid] * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output / "frames" / f"{frame}_depth_mm.png"), depth_mm)
        save_panel(output / "frames" / f"{frame}_overview.png", frame, left, right,
                   disparity, display_depth, args.max_depth)
        tiles.append(contact_tile(left, display_depth, frame, args.max_depth))

        positive = disparity[valid]
        depths = depth[display_valid]
        row = {
            "frame": frame,
            "inference_ms": inference_ms,
            "valid_depth_ratio": float(display_valid.mean()),
            "photometric_error": photometric_error(
                cv2.resize(left, (width, height)), cv2.resize(right, (width, height)), disparity),
            "disparity_median_px": float(np.median(positive)) if positive.size else float("nan"),
            "depth_median_m": float(np.median(depths)) if depths.size else float("nan"),
            "depth_p05_m": float(np.percentile(depths, 5)) if depths.size else float("nan"),
            "depth_p95_m": float(np.percentile(depths, 95)) if depths.size else float("nan"),
        }
        rows.append(row)
        print(json.dumps(row))

    columns = min(4, len(tiles))
    rows_count = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows_count * columns - len(tiles)))
    contact = np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows_count)])
    cv2.imwrite(str(output / "kitti_depth_contact_sheet.png"), contact)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset": str(dataset), "model": str(model), "provider": args.provider,
        "active_providers": session.get_providers(), "frames": len(rows),
        "input_size": [height, width], "normalization": component["normalization"],
        "focal_px": focal, "baseline_m": baseline,
        "mean_inference_ms": float(np.mean([row["inference_ms"] for row in rows])),
        "mean_valid_depth_ratio": float(np.mean([row["valid_depth_ratio"] for row in rows])),
        "mean_photometric_error": float(np.mean([row["photometric_error"] for row in rows])),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
