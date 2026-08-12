#!/usr/bin/env python3
"""Evaluate and plot the fresh EuRoC MH_01 / KITTI 00 loop on-off pair."""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "output" / "loop_benchmark" / "final_pair_20260803"
KITTI_ROOT = Path(os.environ.get("KITTI_DATASET_ROOT", ROOT / "dataset" / "kitti"))
DELTA = 10


def rigid_align(estimate, reference):
    """Return an SE(3)-aligned estimate (rotation + translation, no scale)."""
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


def calculate_metrics(estimate, reference):
    aligned = rigid_align(estimate, reference)
    ape = np.linalg.norm(aligned - reference, axis=1)
    estimate_delta = aligned[DELTA:] - aligned[:-DELTA]
    reference_delta = reference[DELTA:] - reference[:-DELTA]
    rpe = np.linalg.norm(estimate_delta - reference_delta, axis=1)
    return aligned, ape, {
        "samples": int(len(ape)),
        "ape_rmse_m": float(np.sqrt(np.mean(ape**2))),
        "ape_mean_m": float(np.mean(ape)),
        "ape_median_m": float(np.median(ape)),
        "ape_p95_m": float(np.percentile(ape, 95)),
        "ape_max_m": float(np.max(ape)),
        "rpe_translation_rmse_delta10_m": float(np.sqrt(np.mean(rpe**2))),
        "rpe_translation_mean_delta10_m": float(np.mean(rpe)),
    }


def load_csv_trajectory(path):
    data = np.loadtxt(path, delimiter=",", usecols=range(11))
    return data[:, 0], data[:, 1:4]


def load_euroc(name):
    stamps, estimate = load_csv_trajectory(RESULTS / name)
    gt = np.loadtxt(ROOT / "output" / "euroc_MH_01" / "MH01_GT.tum")
    valid = (stamps >= gt[0, 0]) & (stamps <= gt[-1, 0])
    stamps = stamps[valid]
    estimate = estimate[valid]
    reference = np.column_stack(
        [np.interp(stamps, gt[:, 0], gt[:, axis]) for axis in range(1, 4)]
    )
    return stamps, estimate, reference


def load_kitti_matrix(name):
    data = np.loadtxt(RESULTS / name).reshape(-1, 3, 4)
    estimate = data[:, :, 3]
    reference_data = np.loadtxt(KITTI_ROOT / "poses" / "00.txt").reshape(-1, 3, 4)
    count = min(len(estimate), len(reference_data))
    indices = np.arange(count)
    return indices, estimate[:count], reference_data[:count, :, 3]


def load_kitti_optimized(name):
    data = np.loadtxt(RESULTS / name, delimiter=",", usecols=range(11))
    times = np.loadtxt(KITTI_ROOT / "00" / "times.txt")
    reference_data = np.loadtxt(KITTI_ROOT / "poses" / "00.txt").reshape(-1, 3, 4)
    indices = np.searchsorted(times, data[:, 0])
    indices = np.clip(indices, 0, len(times) - 1)
    previous = np.maximum(indices - 1, 0)
    use_previous = (
        np.abs(times[previous] - data[:, 0]) < np.abs(times[indices] - data[:, 0])
    )
    indices[use_previous] = previous[use_previous]

    # KITTIOdomTest records some timestamps both through pubOdometry() and its
    # synchronous post-inputImage() path.  The latter is the newer sample, so
    # retain the final row mapped to each KITTI frame.
    last_row = {}
    for row, index in enumerate(indices):
        last_row[int(index)] = row
    unique_indices = np.array(sorted(last_row), dtype=int)
    rows = np.array([last_row[index] for index in unique_indices], dtype=int)
    return unique_indices, data[rows, 1:4], reference_data[unique_indices, :, 3]


def evaluate(loader, name):
    stamps, estimate, reference = loader(name)
    aligned, ape, result = calculate_metrics(estimate, reference)
    return {
        "stamps": stamps,
        "estimate": estimate,
        "reference": reference,
        "aligned": aligned,
        "ape": ape,
        "metrics": result,
    }


def percent_change(on_value, off_value):
    return 100.0 * (on_value - off_value) / off_value


def plot_comparison(dataset, off, on, axes, output):
    fig, panels = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    colors = {"gt": "#111111", "off": "#d95f02", "on": "#1b9e77"}
    for panel, (axis_a, axis_b, x_label, y_label, equal_aspect) in zip(panels, axes):
        panel.plot(
            off["reference"][:, axis_a], off["reference"][:, axis_b],
            color=colors["gt"], linewidth=2.2, label="Ground truth", zorder=3,
        )
        panel.plot(
            off["aligned"][:, axis_a], off["aligned"][:, axis_b],
            color=colors["off"], linewidth=1.2, label="Loop OFF", alpha=0.9,
        )
        panel.plot(
            on["aligned"][:, axis_a], on["aligned"][:, axis_b],
            color=colors["on"], linewidth=1.2, label="Loop ON (optimized)", alpha=0.9,
        )
        panel.scatter(
            off["reference"][0, axis_a], off["reference"][0, axis_b],
            marker="o", s=45, color="#377eb8", label="Start", zorder=5,
        )
        panel.set_xlabel(f"{x_label} [m]")
        panel.set_ylabel(f"{y_label} [m]")
        panel.grid(True, linewidth=0.5, alpha=0.35)
        if equal_aspect:
            panel.axis("equal")
    panels[0].legend(loc="best", frameon=True)
    fig.suptitle(
        f"{dataset}: SE(3)-aligned trajectories (no scale alignment)\n"
        f"APE RMSE: OFF {off['metrics']['ape_rmse_m']:.4f} m, "
        f"ON {on['metrics']['ape_rmse_m']:.4f} m"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    trajectories = {
        "euroc_mh01_loop_off": evaluate(load_euroc, "euroc_mh01_loop_off.csv"),
        "euroc_mh01_loop_on_online": evaluate(
            load_euroc, "euroc_mh01_loop_on_online.csv"
        ),
        "euroc_mh01_loop_on_optimized": evaluate(
            load_euroc, "euroc_mh01_loop_on_optimized.csv"
        ),
        "kitti00_loop_off": evaluate(load_kitti_matrix, "kitti00_loop_off.txt"),
        "kitti00_loop_on_online": evaluate(
            load_kitti_matrix, "kitti00_loop_on_online.txt"
        ),
        "kitti00_loop_on_optimized": evaluate(
            load_kitti_optimized, "kitti00_loop_on_optimized.csv"
        ),
    }
    report = {name: value["metrics"] for name, value in trajectories.items()}
    euroc_off = trajectories["euroc_mh01_loop_off"]
    euroc_on = trajectories["euroc_mh01_loop_on_optimized"]
    euroc_off_on_on_stamps = np.column_stack([
        np.interp(
            euroc_on["stamps"], euroc_off["stamps"], euroc_off["estimate"][:, axis]
        )
        for axis in range(3)
    ])
    _, _, common_off_metrics = calculate_metrics(
        euroc_off_on_on_stamps, euroc_on["reference"]
    )
    report["euroc_mh01_loop_off_resampled_on_loop_on_stamps"] = common_off_metrics
    for dataset in ("euroc_mh01", "kitti00"):
        off = (
            report["euroc_mh01_loop_off_resampled_on_loop_on_stamps"]
            if dataset == "euroc_mh01"
            else report[f"{dataset}_loop_off"]
        )
        on = report[f"{dataset}_loop_on_optimized"]
        report[f"{dataset}_loop_on_vs_off_percent"] = {
            "ape_rmse": percent_change(on["ape_rmse_m"], off["ape_rmse_m"]),
            "rpe_translation_rmse_delta10": percent_change(
                on["rpe_translation_rmse_delta10_m"],
                off["rpe_translation_rmse_delta10_m"],
            ),
        }

    with (RESULTS / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")

    metric_fields = [
        "samples", "ape_rmse_m", "ape_mean_m", "ape_median_m",
        "ape_p95_m", "ape_max_m", "rpe_translation_rmse_delta10_m",
        "rpe_translation_mean_delta10_m",
    ]
    with (RESULTS / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["trajectory", *metric_fields])
        writer.writeheader()
        for name in trajectories:
            writer.writerow({"trajectory": name, **report[name]})

    plot_comparison(
        "EuRoC MH_01",
        trajectories["euroc_mh01_loop_off"],
        trajectories["euroc_mh01_loop_on_optimized"],
        ((0, 1, "X", "Y", True), (0, 2, "X", "Z", True)),
        RESULTS / "euroc_mh01_trajectory_comparison.png",
    )
    plot_comparison(
        "KITTI 00",
        trajectories["kitti00_loop_off"],
        trajectories["kitti00_loop_on_optimized"],
        ((0, 2, "X", "Z", True), (0, 1, "X", "Y", False)),
        RESULTS / "kitti00_trajectory_comparison.png",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
