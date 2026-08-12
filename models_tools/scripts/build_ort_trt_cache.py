#!/usr/bin/env python3
"""Build ONNX Runtime TensorRT caches compatible with the current C++ runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping

import onnxruntime as ort

from common import load_config, random_inputs, resolve_path, select_components, tensor_shapes


def shape_string(shapes: Mapping[str, tuple[int, ...]]) -> str:
    return ",".join(f"{name}:{'x'.join(map(str, shape))}" for name, shape in shapes.items())


def build_cache(config, name: str) -> None:
    component = config["components"][name]
    onnx_path = resolve_path(config, component["onnx_path"], required=True)
    trt_conf = component.get("tensorrt", {})
    cache_path = resolve_path(config, trt_conf["ort_cache_path"])
    cache_path.mkdir(parents=True, exist_ok=True)
    precision = str(trt_conf.get("precision", config.get("tensorrt", {}).get("precision", "fp16"))).lower()
    options: Dict[str, object] = {
        "trt_fp16_enable": precision == "fp16",
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_path),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_path / "timing.cache"),
        "trt_max_workspace_size": int(
            trt_conf.get("workspace_mib", config.get("tensorrt", {}).get("workspace_mib", 4096))
        ) << 20,
        "trt_builder_optimization_level": int(
            trt_conf.get("builder_optimization_level", config.get("tensorrt", {}).get("builder_optimization_level", 3))
        ),
        "trt_min_subgraph_size": int(trt_conf.get("min_subgraph_size", 1)),
    }
    profiles = trt_conf.get("profiles")
    if profiles:
        minimum = tensor_shapes(component, "min")
        optimum = tensor_shapes(component, "opt")
        maximum = tensor_shapes(component, "max")
        options.update(
            {
                "trt_profile_min_shapes": shape_string(minimum),
                "trt_profile_opt_shapes": shape_string(optimum),
                "trt_profile_max_shapes": shape_string(maximum),
            }
        )
    providers = [("TensorrtExecutionProvider", options)]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_providers = session.get_providers()
    if "TensorrtExecutionProvider" not in active_providers:
        raise RuntimeError(
            "TensorRT provider was requested but is not active; check TensorRT/CUDA "
            f"shared libraries and LD_LIBRARY_PATH (active={active_providers})"
        )
    opt_shapes = tensor_shapes(component, "opt") if profiles else {
        item.name: tuple(int(dim) for dim in item.shape) for item in session.get_inputs()
    }
    inputs = random_inputs(session, opt_shapes, component)
    session.run(None, inputs)
    engines = sorted(cache_path.glob("*.engine"))
    if not engines:
        raise RuntimeError(f"TensorRT ran without producing an engine cache in {cache_path}")
    print(f"built ORT TensorRT cache for {name}: {cache_path} ({len(engines)} engine file(s))")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all", help="component name, comma list, or all")
    args = parser.parse_args()
    config = load_config(args.config)
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime does not provide TensorrtExecutionProvider")
    for name in select_components(config, args.component):
        build_cache(config, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
