#!/usr/bin/env python3
"""Evaluate the checked-in loop benchmark outputs with rigid (no-scale) alignment."""

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "loop_benchmark"
KITTI_ROOT = Path(os.environ.get("KITTI_DATASET_ROOT", ROOT / "dataset" / "kitti"))


def rigid_align(estimate, reference):
    estimate_center = estimate.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (estimate - estimate_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = reference_center - rotation @ estimate_center
    return (rotation @ estimate.T).T + translation


def metrics(estimate, reference, delta=10):
    aligned = rigid_align(estimate, reference)
    ape = np.linalg.norm(aligned - reference, axis=1)
    relative_error = np.linalg.norm(
        (aligned[delta:] - aligned[:-delta])
        - (reference[delta:] - reference[:-delta]), axis=1)
    return {
        "samples": int(len(ape)),
        "ape_rmse_m": float(np.sqrt(np.mean(ape ** 2))),
        "ape_mean_m": float(np.mean(ape)),
        "ape_median_m": float(np.median(ape)),
        "ape_p95_m": float(np.percentile(ape, 95)),
        "ape_max_m": float(np.max(ape)),
        "aligned_end_error_m": float(ape[-1]),
        "rpe_translation_rmse_delta10_m": float(np.sqrt(np.mean(relative_error ** 2))),
    }


def load_euroc_ground_truth():
    data = np.loadtxt(ROOT / "output" / "euroc_MH_01" / "MH01_GT.tum")
    return data[:, 0], data[:, 1:4]


def load_csv_positions(path):
    data = np.loadtxt(path, delimiter=",", usecols=range(11))
    return data[:, 0], data[:, 1:4]


def interpolate_reference(stamps, gt_stamps, gt_positions):
    valid = (stamps >= gt_stamps[0]) & (stamps <= gt_stamps[-1])
    stamps = stamps[valid]
    positions = np.column_stack([
        np.interp(stamps, gt_stamps, gt_positions[:, axis]) for axis in range(3)
    ])
    return valid, positions


def evaluate_euroc(path):
    stamps, positions = load_csv_positions(path)
    gt_stamps, gt_positions = load_euroc_ground_truth()
    valid, reference = interpolate_reference(stamps, gt_stamps, gt_positions)
    return metrics(positions[valid], reference)


def evaluate_euroc_graph(path):
    data = np.loadtxt(path)
    gt_stamps, gt_positions = load_euroc_ground_truth()
    valid, reference = interpolate_reference(data[:, 0], gt_stamps, gt_positions)
    return metrics(data[valid, 1:4], reference)


def load_kitti_ground_truth():
    matrices = np.loadtxt(KITTI_ROOT / "poses" / "00.txt").reshape(-1, 3, 4)
    return matrices[:, :, 3]


def evaluate_kitti(path):
    matrices = np.loadtxt(path).reshape(-1, 3, 4)
    reference = load_kitti_ground_truth()
    count = min(len(matrices), len(reference))
    return metrics(matrices[:count, :, 3], reference[:count])


def evaluate_kitti_graph(path):
    data = np.loadtxt(path)
    times = np.loadtxt(KITTI_ROOT / "00" / "times.txt")
    reference = load_kitti_ground_truth()
    indices = np.searchsorted(times, data[:, 0])
    indices = np.clip(indices, 0, len(times) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = np.abs(times[previous] - data[:, 0]) < np.abs(times[indices] - data[:, 0])
    indices[use_previous] = previous[use_previous]
    return metrics(data[:, 1:4], reference[indices])


def main():
    report = {
        "euroc_mh01_loop_off": evaluate_euroc(RESULTS / "euroc_mh01_loop_off.csv"),
        "euroc_mh01_loop_on_online": evaluate_euroc(RESULTS / "euroc_mh01_loop_on.csv"),
        "euroc_mh01_loop_on_pose_graph": evaluate_euroc_graph(
            RESULTS / "euroc_mh01_pose_graph_on.tum"
        ),
        "kitti00_loop_off": evaluate_kitti(RESULTS / "kitti00_loop_off.txt"),
        "kitti00_loop_on_online": evaluate_kitti(RESULTS / "kitti00_loop_on.txt"),
        "kitti00_loop_on_pose_graph": evaluate_kitti_graph(
            RESULTS / "kitti00_pose_graph_on.tum"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
