#!/usr/bin/env python3
import argparse
import re


POSE_RE = re.compile(
    r"time:\s*([0-9]+(?:\.[0-9]+)?),\s*"
    r"t:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"q:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract VINS console pose lines into TUM format."
    )
    parser.add_argument("input_log")
    parser.add_argument("output_tum")
    args = parser.parse_args()

    poses = []
    with open(args.input_log, "r", encoding="utf-8", errors="replace") as source:
        for line in source:
            match = POSE_RE.search(line)
            if match:
                poses.append(tuple(float(value) for value in match.groups()))

    with open(args.output_tum, "w", encoding="utf-8") as output:
        for pose in poses:
            output.write(
                f"{pose[0]:.9f} "
                + " ".join(f"{value:.9f}" for value in pose[1:])
                + "\n"
            )

    print(f"wrote {len(poses)} poses to {args.output_tum}")
    return 0 if poses else 1


if __name__ == "__main__":
    raise SystemExit(main())
