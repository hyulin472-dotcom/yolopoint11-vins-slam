#!/usr/bin/env python3
"""Copy configured source checkpoints into models_tools and verify checksums."""

from __future__ import annotations

import argparse
import shutil

from common import ensure_parent, load_config, resolve_path, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    entries = config.get("weights", [])
    if not entries:
        print("configuration has no weight copy entries")
        return 0
    for entry in entries:
        source_value = entry.get("source", "")
        if not source_value:
            raise FileNotFoundError(
                f"source checkpoint for {entry.get('name', 'unnamed')} is not configured; "
                "set the environment variable documented in the YAML file"
            )
        source = resolve_path(config, source_value, required=True)
        destination = resolve_path(config, entry["destination"])
        expected = str(entry.get("sha256", "")).lower()
        source_hash = sha256(source)
        if expected and source_hash != expected:
            raise RuntimeError(f"source checksum mismatch for {source}: {source_hash} != {expected}")
        if destination.exists() and not args.force:
            destination_hash = sha256(destination)
            if destination_hash != source_hash:
                raise RuntimeError(f"destination exists with a different checksum: {destination}")
            print(f"verified existing weight: {destination}")
            continue
        ensure_parent(destination)
        shutil.copy2(source, destination)
        destination_hash = sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(f"copy verification failed: {destination}")
        print(f"copied weight: {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
