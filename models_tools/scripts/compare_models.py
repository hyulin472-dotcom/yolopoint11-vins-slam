#!/usr/bin/env python3
"""Compare configured ONNX artifacts with the models currently used by VINS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort

from common import load_config, random_inputs, resolve_path, select_components, sha256, tensor_shapes


def dimensions(value) -> list[Any]:
    return [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim]


def signature(path: Path) -> Dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    return {
        "opset": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": {
            item.name: {
                "dtype": onnx.helper.tensor_dtype_to_string(item.type.tensor_type.elem_type),
                "shape": dimensions(item),
            }
            for item in model.graph.input if item.name not in initializers
        },
        "outputs": {
            item.name: {
                "dtype": onnx.helper.tensor_dtype_to_string(item.type.tensor_type.elem_type),
                "shape": dimensions(item),
            }
            for item in model.graph.output
        },
    }


def signatures_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Treat differently named symbolic dimensions as equivalent dynamics."""
    if left["opset"] != right["opset"]:
        return False
    for section in ("inputs", "outputs"):
        if left[section].keys() != right[section].keys():
            return False
        for name in left[section]:
            first, second = left[section][name], right[section][name]
            if first["dtype"] != second["dtype"] or len(first["shape"]) != len(second["shape"]):
                return False
            for a, b in zip(first["shape"], second["shape"]):
                if isinstance(a, int) and isinstance(b, int) and a != b:
                    return False
                if isinstance(a, int) != isinstance(b, int):
                    return False
    return True


def numerical_compare(generated: Path, reference: Path, shapes: Dict[str, tuple[int, ...]], component) -> Dict[str, Any]:
    generated_session = ort.InferenceSession(str(generated), providers=["CPUExecutionProvider"])
    reference_session = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
    inputs = random_inputs(reference_session, shapes, component, seed=20260804)
    generated_outputs = generated_session.run(None, inputs)
    reference_outputs = reference_session.run(None, inputs)
    report: Dict[str, Any] = {}
    for index, (actual, expected) in enumerate(zip(generated_outputs, reference_outputs)):
        name = reference_session.get_outputs()[index].name
        if np.issubdtype(expected.dtype, np.integer):
            report[name] = {"exact_agreement": float(np.mean(actual == expected))}
            continue
        difference = actual.astype(np.float64) - expected.astype(np.float64)
        denominator = max(float(np.linalg.norm(expected)), 1e-12)
        report[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference) / denominator),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all")
    parser.add_argument("--no-numerical", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    report: Dict[str, Any] = {}
    for name in select_components(config, args.component):
        component = config["components"][name]
        generated = resolve_path(config, component["onnx_path"], required=True)
        reference = resolve_path(config, component["reference_onnx"], required=True)
        generated_signature = signature(generated)
        reference_signature = signature(reference)
        item: Dict[str, Any] = {
            "generated": str(generated),
            "reference": str(reference),
            "generated_size": generated.stat().st_size,
            "reference_size": reference.stat().st_size,
            "generated_sha256": sha256(generated),
            "reference_sha256": sha256(reference),
            "signature_equal": generated_signature == reference_signature,
            "signature_compatible": signatures_compatible(generated_signature, reference_signature),
            "generated_signature": generated_signature,
            "reference_signature": reference_signature,
        }
        if not args.no_numerical:
            shapes = tensor_shapes(component, "compare")
            item["numerical"] = numerical_compare(generated, reference, shapes, component)
        report[name] = item
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        output_path = resolve_path(config, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
