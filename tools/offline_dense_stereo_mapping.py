#!/usr/bin/env python3
import argparse
import bisect
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


# Keep this preprocessing identical to OnlineDenseMapper::imageToTensor().
# The deployed ONNX graph has its internal normalization stripped.
FOUNDATION_IMAGENET_MEAN = np.asarray([123.675, 116.280, 103.530], dtype=np.float32).reshape(1, 3, 1, 1)
FOUNDATION_IMAGENET_STD = np.asarray([58.395, 57.120, 57.375], dtype=np.float32).reshape(1, 3, 1, 1)


class OpenCVYamlLoader(yaml.SafeLoader):
    pass


def opencv_matrix_constructor(loader, node):
    return loader.construct_mapping(node, deep=True)


OpenCVYamlLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def load_opencv_yaml(path):
    text = open(path, "r", encoding="utf-8").read()
    text = text.replace("%YAML:1.0", "")
    return yaml.load(text, Loader=OpenCVYamlLoader)


def read_general_config(path):
    cfg = load_opencv_yaml(path)
    if cfg is None:
        raise RuntimeError(f"empty config: {path}")
    return cfg


def opencv_matrix_to_np(node):
    data = np.asarray(node["data"], dtype=np.float64)
    return data.reshape((int(node["rows"]), int(node["cols"])))


def read_camera_intrinsic(config_dir, calib_name):
    cam = load_opencv_yaml(os.path.join(config_dir, calib_name))
    proj = cam["projection_parameters"]
    dist = cam.get("distortion_parameters", {})
    mirror = cam.get("mirror_parameters", {})
    k = np.array(
        [[proj["gamma1"], 0.0, proj["u0"]], [0.0, proj["gamma2"], proj["v0"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    d = np.array(
        [dist.get("k1", 0.0), dist.get("k2", 0.0), dist.get("p1", 0.0), dist.get("p2", 0.0)],
        dtype=np.float64,
    )
    xi = float(mirror.get("xi", 0.0))
    model_type = str(cam.get("model_type", "PINHOLE")).upper()
    return k, d, xi, model_type


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_rot(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_tum_trajectory(path):
    traj = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) < 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = vals[:8]
            traj.append((t, np.array([tx, ty, tz], dtype=np.float64), quat_to_rot(qx, qy, qz, qw)))
    if not traj:
        raise RuntimeError(f"no trajectory poses loaded from {path}")
    return sorted(traj, key=lambda x: x[0])


def nearest_pose(traj, t, max_dt):
    times = [p[0] for p in traj]
    idx = bisect.bisect_left(times, t)
    best = None
    for j in (idx - 1, idx):
        if 0 <= j < len(traj):
            dt = abs(traj[j][0] - t)
            if best is None or dt < best[0]:
                best = (dt, traj[j])
    if best is None or best[0] > max_dt:
        return None
    return best[1]


def read_bag_topics(bag_uri, storage_id):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topic_types


def collect_images(bag_uri, storage_id, left_topic, right_topic):
    reader, topic_types = read_bag_topics(bag_uri, storage_id)
    if left_topic not in topic_types:
        raise RuntimeError(f"left topic not found in bag: {left_topic}")
    if right_topic not in topic_types:
        raise RuntimeError(f"right topic not found in bag: {right_topic}")
    left_type = get_message(topic_types[left_topic])
    right_type = get_message(topic_types[right_topic])
    bridge = CvBridge()
    left, right = [], []
    while reader.has_next():
        topic, raw, bag_time_ns = reader.read_next()
        if topic != left_topic and topic != right_topic:
            continue
        msg_type = left_type if topic == left_topic else right_type
        msg = deserialize_message(raw, msg_type)
        t = stamp_to_sec(msg.header.stamp)
        if t <= 0.0:
            t = bag_time_ns * 1e-9
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        if topic == left_topic:
            left.append((t, img.copy()))
        else:
            right.append((t, img.copy()))
    return left, right


def sync_stereo(left, right, max_dt):
    right_times = [x[0] for x in right]
    pairs = []
    for t, img in left:
        idx = bisect.bisect_left(right_times, t)
        best = None
        for j in (idx - 1, idx):
            if 0 <= j < len(right):
                dt = abs(right[j][0] - t)
                if best is None or dt < best[0]:
                    best = (dt, right[j])
        if best is not None and best[0] <= max_dt:
            pairs.append((t, img, best[1][1]))
    return pairs


def build_pinhole_rectifier(cfg, config_dir, k0, d0, k1, d1, t_bc0, t_bc1):
    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    t_c0_b = np.linalg.inv(t_bc0)
    t_c0_c1 = t_c0_b @ t_bc1
    r = t_c0_c1[:3, :3]
    t = t_c0_c1[:3, 3]
    image_size = (width, height)
    r0, r1, p0, p1, q, _, _ = cv2.stereoRectify(
        k0, d0, k1, d1, image_size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map00, map01 = cv2.initUndistortRectifyMap(k0, d0, r0, p0, image_size, cv2.CV_32FC1)
    map10, map11 = cv2.initUndistortRectifyMap(k1, d1, r1, p1, image_size, cv2.CV_32FC1)
    baseline = abs(float(p1[0, 3] / p1[0, 0]))
    fx = float(p0[0, 0])
    cx = float(p0[0, 2])
    cy = float(p0[1, 2])
    return (map00, map01, map10, map11), (fx, cx, cy, baseline), "pinhole"


def build_omnidir_rectifier(cfg, k0, d0, xi0, k1, d1, xi1, t_bc0, t_bc1):
    if not hasattr(cv2, "omnidir"):
        raise RuntimeError("this OpenCV build does not provide cv2.omnidir")

    width = int(cfg["image_width"])
    height = int(cfg["image_height"])
    t_c0_b = np.linalg.inv(t_bc0)
    t_c0_c1 = t_c0_b @ t_bc1
    r01 = t_c0_c1[:3, :3]
    t01 = t_c0_c1[:3, 3]
    r = r01.T
    t = (-r01.T @ t01).reshape(3, 1)

    r0, r1 = cv2.omnidir.stereoRectify(r, t)
    focal = float(cfg.get("offline_dense_mapping_rectified_focal", 460.0))
    cx = (width - 1.0) * 0.5
    cy = (height - 1.0) * 0.5
    p = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    image_size = (width, height)

    xi0_mat = np.array([xi0], dtype=np.float64)
    xi1_mat = np.array([xi1], dtype=np.float64)
    map00, map01 = cv2.omnidir.initUndistortRectifyMap(
        k0, d0, xi0_mat, r0, p, image_size, cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    map10, map11 = cv2.omnidir.initUndistortRectifyMap(
        k1, d1, xi1_mat, r1, p, image_size, cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    baseline = float(np.linalg.norm(t))
    return (map00, map01, map10, map11), (focal, cx, cy, baseline), "omnidir"


def build_rectifier(cfg, config_dir):
    k0, d0, xi0, model0 = read_camera_intrinsic(config_dir, cfg["cam0_calib"])
    k1, d1, xi1, model1 = read_camera_intrinsic(config_dir, cfg["cam1_calib"])
    t_bc0 = opencv_matrix_to_np(cfg["body_T_cam0"])
    t_bc1 = opencv_matrix_to_np(cfg["body_T_cam1"])
    mode = str(cfg.get("offline_dense_mapping_rectification_model", "omnidir")).lower()

    if mode == "omnidir" and model0 == "MEI" and model1 == "MEI":
        try:
            rectifier, intr, rect_mode = build_omnidir_rectifier(cfg, k0, d0, xi0, k1, d1, xi1, t_bc0, t_bc1)
            return rectifier, intr, t_bc0, rect_mode
        except Exception as exc:
            print(f"warning: omnidir rectification failed ({exc}); fallback to pinhole rectification")

    rectifier, intr, rect_mode = build_pinhole_rectifier(cfg, config_dir, k0, d0, k1, d1, t_bc0, t_bc1)
    return rectifier, intr, t_bc0, rect_mode


def create_matcher(width, max_disparity, min_disparity=0):
    num_disp = max(16, int(max_disparity))
    num_disp = int(math.ceil(num_disp / 16.0) * 16)
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def save_debug_images(output_ply, left_rect, right_rect, disparity, valid):
    stem = os.path.splitext(expand_path(output_ply))[0]
    debug_dir = stem + "_debug"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "left_rect.png"), left_rect)
    cv2.imwrite(os.path.join(debug_dir, "right_rect.png"), right_rect)
    disp_vis = disparity.copy()
    disp_vis[~np.isfinite(disp_vis)] = 0.0
    disp_vis[disp_vis < 0.0] = 0.0
    if np.max(disp_vis) > 0.0:
        disp_vis = np.clip(disp_vis / np.percentile(disp_vis[disp_vis > 0.0], 95) * 255.0, 0.0, 255.0)
    cv2.imwrite(os.path.join(debug_dir, "disparity.png"), disp_vis.astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, "valid_mask.png"), (valid.astype(np.uint8) * 255))
    print(f"debug images: {debug_dir}")


def compute_disparity_and_mask(left_rect, right_rect, matcher, cfg):
    min_disp = float(cfg.get("offline_dense_mapping_min_disparity", 2.0))
    max_disp = float(cfg.get("offline_dense_mapping_max_disparity", 96.0))
    lr_check = bool(cfg.get("offline_dense_mapping_lr_check", True))
    lr_max_diff = float(cfg.get("offline_dense_mapping_lr_max_diff", 1.5))
    min_gradient = float(cfg.get("offline_dense_mapping_min_gradient", 12.0))

    disp_left = matcher.compute(left_rect, right_rect).astype(np.float32) / 16.0
    valid = disp_left >= min_disp
    valid &= disp_left <= max_disp

    if lr_check:
        right_matcher = create_matcher(left_rect.shape[1], max_disp, min_disparity=-int(math.ceil(max_disp / 16.0) * 16))
        disp_right = right_matcher.compute(right_rect, left_rect).astype(np.float32) / 16.0
        h, w = disp_left.shape
        xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        ys = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        right_x = xs - disp_left
        sampled_right = cv2.remap(disp_right, right_x, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid &= np.abs(disp_left + sampled_right) <= lr_max_diff

    if min_gradient > 0.0:
        grad_x = cv2.Sobel(left_rect, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(left_rect, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        valid &= grad >= min_gradient

    return disp_left, valid


class FoundationStereoDepthSource:
    def __init__(self, cfg):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for offline_dense_mapping_depth_source=foundation_stereo. "
                "Run this script with your torch128 environment or switch depth_source to sgbm."
            ) from exc

        self.ort = ort
        self.model_path = expand_path(str(cfg.get("offline_dense_mapping_foundation_model_path")))
        self.input_width = int(cfg.get("offline_dense_mapping_foundation_input_width", 736))
        self.input_height = int(cfg.get("offline_dense_mapping_foundation_input_height", 320))
        use_tensorrt = bool(cfg.get("offline_dense_mapping_foundation_use_tensorrt", False))
        use_cuda = bool(cfg.get("offline_dense_mapping_foundation_use_cuda", True))
        providers = []
        available = ort.get_available_providers()
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            cache_path = expand_path(
                str(cfg.get("offline_dense_mapping_foundation_trt_cache_path", "onxx/fast_foundationstereo/trt_cache"))
            )
            os.makedirs(cache_path, exist_ok=True)
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache_path,
                    },
                )
            )
        if use_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.left_name = self.session.get_inputs()[0].name
        self.right_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"foundation stereo providers: {self.session.get_providers()}")

    def _preprocess(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, :, :, :]
        return np.ascontiguousarray((tensor - FOUNDATION_IMAGENET_MEAN) / FOUNDATION_IMAGENET_STD)

    def infer_disparity(self, left_rect, right_rect):
        h, w = left_rect.shape[:2]
        left_tensor = self._preprocess(left_rect)
        right_tensor = self._preprocess(right_rect)
        output = self.session.run(
            [self.output_name],
            {self.left_name: left_tensor, self.right_name: right_tensor},
        )[0]
        output = np.asarray(output, dtype=np.float32)
        if output.ndim < 2:
            raise RuntimeError(f"unexpected FoundationStereo output shape: {output.shape}")
        output_height, output_width = output.shape[-2:]
        disp = output.reshape(-1, output_height, output_width)[0]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        disp *= float(w) / float(output_width)
        return disp


def compute_foundation_disparity_and_mask(left_rect, right_rect, depth_source, cfg):
    disp = depth_source.infer_disparity(left_rect, right_rect)
    min_disp = float(cfg.get("offline_dense_mapping_min_disparity", 2.0))
    max_disp = float(cfg.get("offline_dense_mapping_max_disparity", 96.0))
    min_gradient = float(cfg.get("offline_dense_mapping_min_gradient", 12.0))
    valid = np.isfinite(disp)
    valid &= disp >= min_disp
    valid &= disp <= max_disp
    if min_gradient > 0.0:
        grad_x = cv2.Sobel(left_rect, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(left_rect, cv2.CV_32F, 0, 1, ksize=3)
        valid &= cv2.magnitude(grad_x, grad_y) >= min_gradient
    return disp, valid


def raycast_voxels(start, end, resolution):
    start_idx = np.floor(start / resolution).astype(np.int64)
    end_idx = np.floor(end / resolution).astype(np.int64)
    diff = end_idx - start_idx
    steps = int(np.max(np.abs(diff)))
    if steps <= 0:
        return [], tuple(end_idx.tolist())
    free = []
    for i in range(steps):
        alpha = float(i) / float(steps)
        idx = np.rint(start_idx.astype(np.float64) + alpha * diff.astype(np.float64)).astype(np.int64)
        free.append(tuple(idx.tolist()))
    return free, tuple(end_idx.tolist())


def update_occupancy(logodds, cam_origin, points_world, resolution, pixel_step, free_inc, occ_inc):
    if points_world.shape[0] == 0:
        return
    sampled = points_world[::max(1, int(pixel_step))]
    for point in sampled:
        free_voxels, occupied_voxel = raycast_voxels(cam_origin, point, resolution)
        for voxel in free_voxels:
            logodds[voxel] += free_inc
        logodds[occupied_voxel] += occ_inc


def save_occupancy_outputs(logodds, cfg):
    output_dir = expand_path(str(cfg.get("offline_dense_mapping_occupancy_output_dir", "~/output/offline_occupancy_map")))
    os.makedirs(output_dir, exist_ok=True)
    resolution = float(cfg.get("offline_dense_mapping_occupancy_resolution", 0.10))
    free_threshold = float(cfg.get("offline_dense_mapping_occupancy_free_threshold", -0.02))
    occupied_threshold = float(cfg.get("offline_dense_mapping_occupancy_occupied_threshold", 0.08))

    if not logodds:
        print("occupancy: no voxels generated")
        return

    keys = np.asarray(list(logodds.keys()), dtype=np.int64)
    values = np.asarray([logodds[tuple(k)] for k in keys], dtype=np.float32)
    min_idx = keys.min(axis=0)
    max_idx = keys.max(axis=0)
    shape = (max_idx - min_idx + 1).astype(np.int64)
    grid = np.zeros(tuple(shape.tolist()), dtype=np.uint8)
    local = keys - min_idx.reshape(1, 3)
    grid[local[values < free_threshold, 0], local[values < free_threshold, 1], local[values < free_threshold, 2]] = 1
    grid[local[values > occupied_threshold, 0], local[values > occupied_threshold, 1], local[values > occupied_threshold, 2]] = 2

    origin = min_idx.astype(np.float32) * resolution
    meta = np.array([origin[0], origin[1], origin[2], resolution], dtype=np.float32)
    np.save(os.path.join(output_dir, "occupancy_grid.npy"), grid)
    np.save(os.path.join(output_dir, "occupancy_meta.npy"), meta)

    xy = np.max(grid, axis=2)
    image = np.zeros_like(xy, dtype=np.uint8)
    image[xy == 1] = 127
    image[xy == 2] = 255
    cv2.imwrite(os.path.join(output_dir, "occupancy_2d_image.png"), image)

    occ_idx = np.argwhere(grid == 2)
    if occ_idx.size > 0:
        centers = (occ_idx.astype(np.float32) + min_idx.reshape(1, 3).astype(np.float32) + 0.5) * resolution
        colors = np.tile(np.array([[255, 80, 40]], dtype=np.uint8), (centers.shape[0], 1))
        write_ply(os.path.join(output_dir, "occupied_points.ply"), centers, colors)

    print(f"occupancy output: {output_dir}")
    print(f"occupancy grid shape: {grid.shape}, occupied voxels: {int(np.count_nonzero(grid == 2))}, free voxels: {int(np.count_nonzero(grid == 1))}")


def voxel_downsample(points, colors, voxel):
    if voxel <= 0.0 or len(points) == 0:
        return points, colors
    buckets = {}
    for p, c in zip(points, colors):
        key = tuple(np.floor(p / voxel).astype(np.int64).tolist())
        if key not in buckets:
            buckets[key] = (p, c)
    pts = np.asarray([v[0] for v in buckets.values()], dtype=np.float32)
    cols = np.asarray([v[1] for v in buckets.values()], dtype=np.uint8)
    return pts, cols


def write_ply(path, points, colors):
    parent = os.path.dirname(expand_path(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    path = expand_path(path)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("ply\nformat ascii 1.0\n")
        fout.write(f"element vertex {len(points)}\n")
        fout.write("property float x\nproperty float y\nproperty float z\n")
        fout.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fout.write("end_header\n")
        for p, c in zip(points, colors):
            fout.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def run_mapping(args):
    cfg = read_general_config(args.config)
    if not bool(cfg.get("offline_dense_mapping_enable", False)) and not args.force:
        print("offline_dense_mapping_enable is false; nothing to do. Use --force to override.")
        return 0

    config_dir = os.path.dirname(os.path.abspath(args.config))
    output_ply = args.output or cfg.get("offline_dense_mapping_output_ply", "~/output/offline_dense_map.ply")
    stride = args.stride if args.stride is not None else int(cfg.get("offline_dense_mapping_stride", 10))
    min_depth = float(cfg.get("offline_dense_mapping_min_depth", 0.3))
    max_depth = float(cfg.get("offline_dense_mapping_max_depth", 20.0))
    voxel = float(cfg.get("offline_dense_mapping_voxel_size", 0.05))
    max_points_per_frame = int(cfg.get("offline_dense_mapping_max_points_per_frame", 8000))
    pixel_step = max(1, int(cfg.get("offline_dense_mapping_pixel_step", 1)))
    save_debug = bool(cfg.get("offline_dense_mapping_save_debug", False))
    max_disparity = float(cfg.get("offline_dense_mapping_max_disparity", 96.0))
    depth_source_name = str(cfg.get("offline_dense_mapping_depth_source", "sgbm")).lower()
    use_occupancy = bool(cfg.get("offline_dense_mapping_use_occupancy", False))
    occupancy_resolution = float(cfg.get("offline_dense_mapping_occupancy_resolution", 0.10))
    occupancy_pixel_step = int(cfg.get("offline_dense_mapping_occupancy_pixel_step", 12))
    occupancy_free_inc = float(cfg.get("offline_dense_mapping_occupancy_free_logodds", -0.05))
    occupancy_occ_inc = float(cfg.get("offline_dense_mapping_occupancy_occupied_logodds", 0.20))

    left, right = collect_images(args.bag, args.storage_id, cfg["image0_topic"], cfg["image1_topic"])
    stereo = sync_stereo(left, right, args.sync_tolerance)
    traj = load_tum_trajectory(args.trajectory)
    rectifier, intr, t_bc0, rectification_mode = build_rectifier(cfg, config_dir)
    map00, map01, map10, map11 = rectifier
    fx, cx, cy, baseline = intr
    t_bc0_r = t_bc0[:3, :3]
    t_bc0_t = t_bc0[:3, 3]
    matcher = create_matcher(int(cfg["image_width"]), max_disparity)
    depth_source = None
    if depth_source_name == "foundation_stereo":
        depth_source = FoundationStereoDepthSource(cfg)
    elif depth_source_name != "sgbm":
        raise RuntimeError(f"unknown offline_dense_mapping_depth_source: {depth_source_name}")

    all_points, all_colors = [], []
    occupancy_logodds = defaultdict(float)
    used_frames = 0
    for i, (t, left_img, right_img) in enumerate(stereo):
        if i % stride != 0:
            continue
        pose = nearest_pose(traj, t, args.pose_tolerance)
        if pose is None:
            continue
        left_rect = cv2.remap(left_img, map00, map01, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, map10, map11, cv2.INTER_LINEAR)
        if depth_source_name == "foundation_stereo":
            disparity, valid = compute_foundation_disparity_and_mask(left_rect, right_rect, depth_source, cfg)
        else:
            disparity, valid = compute_disparity_and_mask(left_rect, right_rect, matcher, cfg)
        if save_debug:
            save_debug_images(output_ply, left_rect, right_rect, disparity, valid)
            save_debug = False
        depth = fx * baseline / np.maximum(disparity, 1e-6)
        valid &= np.isfinite(depth)
        valid &= depth > min_depth
        valid &= depth < max_depth

        # Match the online mapper's regular pixel-grid sampling before applying
        # the per-frame point cap.
        if pixel_step > 1:
            sample_mask = np.zeros_like(valid, dtype=bool)
            sample_mask[::pixel_step, ::pixel_step] = True
            valid &= sample_mask

        ys, xs = np.where(valid)
        if len(xs) == 0:
            continue
        if len(xs) > max_points_per_frame:
            idx = np.linspace(0, len(xs) - 1, max_points_per_frame).astype(np.int64)
            xs, ys = xs[idx], ys[idx]

        z = depth[ys, xs]
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fx
        pts_cam0 = np.stack([x, y, z], axis=1)

        _, p_wb, r_wb = pose
        pts_body = (t_bc0_r @ pts_cam0.T).T + t_bc0_t.reshape(1, 3)
        pts_world = (r_wb @ pts_body.T).T + p_wb.reshape(1, 3)
        cam_origin_world = r_wb @ t_bc0_t + p_wb
        if use_occupancy:
            update_occupancy(
                occupancy_logodds,
                cam_origin_world,
                pts_world,
                occupancy_resolution,
                occupancy_pixel_step,
                occupancy_free_inc,
                occupancy_occ_inc,
            )
        gray = left_rect[ys, xs]
        colors = np.stack([gray, gray, gray], axis=1).astype(np.uint8)
        all_points.append(pts_world.astype(np.float32))
        all_colors.append(colors)
        used_frames += 1
        if used_frames % 20 == 0:
            print(f"processed dense frames: {used_frames}, points: {sum(len(p) for p in all_points)}")

    if not all_points:
        raise RuntimeError("no dense points generated")
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    points, colors = voxel_downsample(points, colors, voxel)
    write_ply(output_ply, points, colors)
    if use_occupancy:
        save_occupancy_outputs(occupancy_logodds, cfg)
    print(f"rectification: {rectification_mode}")
    print(f"depth source: {depth_source_name}")
    print(f"rectified fx: {fx:.3f}, baseline: {baseline:.6f}")
    print(f"stereo pairs: {len(stereo)}")
    print(f"used dense frames: {used_frames}")
    print(f"wrote points: {len(points)}")
    print(f"output: {expand_path(output_ply)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Offline dense stereo mapping from a ROS2 stereo bag and a TUM trajectory.")
    parser.add_argument("--config", default="config/offline_dense_mapping/euroc.yaml")
    parser.add_argument("--bag", required=True, help="ROS2 bag containing raw stereo image topics")
    parser.add_argument("--trajectory", required=True, help="TUM trajectory generated from VINS odometry")
    parser.add_argument("--output", default=None, help="Output PLY path; default comes from YAML")
    parser.add_argument("--stride", type=int, default=None, help="Use every Nth stereo pair; default comes from YAML")
    parser.add_argument("--sync-tolerance", type=float, default=0.003)
    parser.add_argument("--pose-tolerance", type=float, default=0.03)
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--force", action="store_true", help="Run even when offline_dense_mapping_enable is false")
    return run_mapping(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
