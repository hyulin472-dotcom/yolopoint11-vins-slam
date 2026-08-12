#!/usr/bin/env python3
"""Check configured ONNX graphs and run selected shape profiles."""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort

from common import load_config, random_inputs, resolve_path, select_components, tensor_shapes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all")
    parser.add_argument("--profiles", default="compare", help="comma-separated TensorRT profile names")
    args = parser.parse_args()
    config = load_config(args.config)
    profile_names = [value.strip() for value in args.profiles.split(",") if value.strip()]
    report = {}
    for name in select_components(config, args.component):
        component = config["components"][name]
        path = resolve_path(config, component["onnx_path"], required=True)
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        runs = []
        for profile_name in profile_names:
            shapes = tensor_shapes(component, profile_name)
            outputs = session.run(None, random_inputs(session, shapes, component, seed=20260804))
            if any(not np.all(np.isfinite(value)) for value in outputs if np.issubdtype(value.dtype, np.floating)):
                raise RuntimeError(f"non-finite output: {name}/{profile_name}")
            runs.append({
                "profile": profile_name,
                "inputs": {key: list(value) for key, value in shapes.items()},
                "outputs": {
                    item.name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for item, value in zip(session.get_outputs(), outputs)
                },
            })
        report[name] = {"onnx_checker": "passed", "runs": runs}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
