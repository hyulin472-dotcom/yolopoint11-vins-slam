#!/usr/bin/env python3
"""Shared configuration helpers for model export and TensorRT tooling."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_environment(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        return "" if default is None else default

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _expand_tree(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_environment(value)
    if isinstance(value, list):
        return [_expand_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_tree(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    config = _expand_tree(config)
    config["_config_path"] = config_path
    configured_root = config.get("project_root", "../..")
    root = Path(configured_root)
    if not root.is_absolute():
        root = config_path.parent / root
    config["_project_root"] = root.resolve()
    return config


def resolve_path(config: Mapping[str, Any], value: str | Path, *, required: bool = False) -> Path:
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError("required path is empty")
        return Path()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shape(shape: Iterable[Any]) -> tuple[int, ...]:
    parsed = tuple(int(value) for value in shape)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError(f"invalid tensor shape: {shape}")
    return parsed


def select_components(config: Mapping[str, Any], requested: str | None) -> list[str]:
    components = config.get("components", {})
    if not isinstance(components, dict) or not components:
        raise ValueError("configuration has no components")
    if not requested or requested == "all":
        return list(components)
    names = [name.strip() for name in requested.split(",") if name.strip()]
    unknown = [name for name in names if name not in components]
    if unknown:
        raise KeyError(f"unknown component(s): {', '.join(unknown)}")
    return names


def tensor_shapes(component: Mapping[str, Any], profile_name: str) -> Dict[str, tuple[int, ...]]:
    profiles = component.get("tensorrt", {}).get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"missing TensorRT {profile_name} profile")
    return {name: parse_shape(shape) for name, shape in profiles[profile_name].items()}


def random_inputs(session: Any, shapes: Mapping[str, tuple[int, ...]], component: Mapping[str, Any], seed: int = 0):
    """Create deterministic, model-appropriate test inputs without fixed dimensions."""
    rng = np.random.default_rng(seed)
    height, width = map(float, component.get("image_size", component.get("input_size", [1, 1])))
    inputs: Dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = tuple(shapes[item.name])
        if "int64" in item.type:
            value = np.zeros(shape, dtype=np.int64)
        elif "int32" in item.type:
            value = np.zeros(shape, dtype=np.int32)
        else:
            value = rng.random(shape, dtype=np.float32)
            if item.name.startswith("image_size") and shape[-1] == 2:
                value[...] = np.array([width, height], dtype=np.float32)
            elif item.name.startswith("keypoints") and shape[-1] == 2:
                value *= np.array([width, height], dtype=np.float32)
            elif item.name.startswith("descriptors"):
                value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)
            elif component.get("normalization") == "imagenet_0_255":
                value *= 255.0
                mean = np.asarray(component["normalization_mean"], dtype=np.float32)
                std = np.asarray(component["normalization_std"], dtype=np.float32)
                value = (value - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
            elif str(component.get("normalization", "")).endswith("_0_255"):
                value *= 255.0
        inputs[item.name] = value
    return inputs
