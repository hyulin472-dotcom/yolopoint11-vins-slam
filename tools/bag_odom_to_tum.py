#!/usr/bin/env python3
import argparse
import os
import sys

import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def export_odom_to_tum(bag_uri, topic, output_path, storage_id):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        while reader.has_next():
            name, raw, bag_time_ns = reader.read_next()
            if name != topic:
                continue

            msg = deserialize_message(raw, Odometry)
            t = stamp_to_sec(msg.header.stamp)
            if t <= 0.0:
                t = bag_time_ns * 1e-9

            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            fout.write(
                f"{t:.9f} "
                f"{p.x:.9f} {p.y:.9f} {p.z:.9f} "
                f"{q.x:.9f} {q.y:.9f} {q.z:.9f} {q.w:.9f}\n"
            )
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export a ROS2 nav_msgs/Odometry topic from rosbag2 to TUM trajectory format."
    )
    parser.add_argument("bag_uri", help="ROS2 bag directory, for example vins_traj_sift")
    parser.add_argument("topic", help="Odometry topic, for example /vins_estimator/odometry")
    parser.add_argument("output_tum", help="Output TUM txt path")
    parser.add_argument(
        "--storage-id",
        default="sqlite3",
        help="rosbag2 storage id, default: sqlite3",
    )
    args = parser.parse_args()

    count = export_odom_to_tum(args.bag_uri, args.topic, args.output_tum, args.storage_id)
    if count == 0:
        print(f"no messages found on topic {args.topic}", file=sys.stderr)
        return 1

    print(f"wrote {count} poses to {args.output_tum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
