#!/usr/bin/env python3
"""Stage checksum-verified prebuilt ONNX files declared by a model YAML."""

from __future__ import annotations

import argparse
import shutil

from common import ensure_parent, load_config, resolve_path, select_components, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for name in select_components(config, args.component):
        component = config["components"][name]
        prebuilt = component.get("prebuilt_onnx")
        if not prebuilt:
            print(f"skip {name}: no prebuilt_onnx entry")
            continue
        source = resolve_path(config, prebuilt["source"], required=True)
        destination = resolve_path(config, component["onnx_path"])
        expected = str(prebuilt.get("sha256", "")).lower()
        actual = sha256(source)
        if expected and actual != expected:
            raise RuntimeError(f"checksum mismatch for {source}: expected {expected}, got {actual}")
        if destination.exists() and not args.force:
            if sha256(destination) != actual:
                raise RuntimeError(f"destination exists with different content: {destination}; use --force")
            print(f"already staged {name}: {destination}")
            continue
        ensure_parent(destination)
        shutil.copy2(source, destination)
        print(f"staged {name}: {source} -> {destination} (sha256={actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
