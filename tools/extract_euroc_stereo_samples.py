#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


def image_to_mono8(message):
    if message.encoding not in ("mono8", "8UC1"):
        raise RuntimeError(f"unsupported image encoding: {message.encoding}")
    array = np.frombuffer(message.data, dtype=np.uint8)
    return array.reshape(message.height, message.step)[:, : message.width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("output")
    parser.add_argument(
        "--ranges",
        default="100:140,900:940,1800:1840,2700:2740,3500:3540",
        help="comma-separated half-open source-frame ranges",
    )
    args = parser.parse_args()

    selected = set()
    for item in args.ranges.split(","):
        begin, end = (int(value) for value in item.split(":"))
        selected.update(range(begin, end))
    os.makedirs(args.output, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_index = {"/cam0/image_raw": 0, "/cam1/image_raw": 0}
    saved = {"/cam0/image_raw": {}, "/cam1/image_raw": {}}
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in topic_index:
            continue
        source_index = topic_index[topic]
        topic_index[topic] += 1
        if source_index not in selected:
            continue
        message = deserialize_message(raw, Image)
        image = image_to_mono8(message)
        saved[topic][source_index] = image.copy()

    common = sorted(
        set(saved["/cam0/image_raw"]) & set(saved["/cam1/image_raw"])
    )
    mapping_path = os.path.join(args.output, "source_indices.txt")
    with open(mapping_path, "w", encoding="utf-8") as mapping:
        for output_index, source_index in enumerate(common):
            left_path = os.path.join(args.output, f"left_{output_index:03d}.png")
            right_path = os.path.join(args.output, f"right_{output_index:03d}.png")
            cv2.imwrite(left_path, saved["/cam0/image_raw"][source_index])
            cv2.imwrite(right_path, saved["/cam1/image_raw"][source_index])
            mapping.write(f"{output_index} {source_index}\n")

    print(f"saved {len(common)} synchronized stereo pairs to {args.output}")
    return 0 if common else 1


if __name__ == "__main__":
    raise SystemExit(main())
