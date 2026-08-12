#!/usr/bin/env python3
"""Build standalone TensorRT engines from configured ONNX components."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_parent, load_config, resolve_path, select_components, tensor_shapes


def build_engine(config, name: str, force: bool) -> None:
    import tensorrt as trt

    component = config["components"][name]
    onnx_path = resolve_path(config, component["onnx_path"], required=True)
    trt_conf = component.get("tensorrt", {})
    if not bool(trt_conf.get("standalone", True)):
        print(f"skip standalone engine for {name}: use the ONNX Runtime TensorRT cache")
        return
    engine_path = resolve_path(config, trt_conf["engine_path"])
    if engine_path.exists() and not force:
        print(f"skip existing engine: {engine_path}")
        return
    ensure_parent(engine_path)

    severity_name = str(config.get("tensorrt", {}).get("log_level", "warning")).upper()
    severity = getattr(trt.Logger, severity_name, trt.Logger.WARNING)
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    # parse_from_file also resolves ONNX external-data sidecars (for example
    # the large LightGlue weight file emitted by the dynamo exporter).
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    build_config = builder.create_builder_config()
    precision = str(trt_conf.get("precision", config.get("tensorrt", {}).get("precision", "fp16"))).lower()
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("warning: platform_has_fast_fp16 is false; engine will still be requested as FP16")
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"precision must be fp32 or fp16, got {precision}")

    workspace_mib = int(trt_conf.get("workspace_mib", config.get("tensorrt", {}).get("workspace_mib", 4096)))
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    optimization_level = int(
        trt_conf.get("builder_optimization_level", config.get("tensorrt", {}).get("builder_optimization_level", 3))
    )
    if hasattr(build_config, "builder_optimization_level"):
        build_config.builder_optimization_level = optimization_level

    dynamic_inputs = any(-1 in tuple(network.get_input(i).shape) for i in range(network.num_inputs))
    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        minimum = tensor_shapes(component, "min")
        optimum = tensor_shapes(component, "opt")
        maximum = tensor_shapes(component, "max")
        for index in range(network.num_inputs):
            input_tensor = network.get_input(index)
            input_name = input_tensor.name
            if input_name not in minimum or input_name not in optimum or input_name not in maximum:
                raise KeyError(f"dynamic input {input_name} is missing from one or more profiles")
            profile.set_shape(input_name, minimum[input_name], optimum[input_name], maximum[input_name])
        build_config.add_optimization_profile(profile)

    print(f"building {name}: precision={precision}, workspace={workspace_mib} MiB")
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed: {name}")
    engine_path.write_bytes(serialized)
    print(f"wrote engine: {engine_path} ({engine_path.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all", help="component name, comma list, or all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in select_components(config, args.component):
        build_engine(config, name, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
