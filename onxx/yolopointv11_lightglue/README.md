# YOLOPointv11 + LightGlue ONNX models

These are standalone deployment models for the VINS `feature_tracker_type: 3`
frontend. ONNX Runtime does not need the `glue-factory` Python package or source
tree to load or execute them.

## Provenance

- Source experiment: `yolopointv11+lg_homography`
- Source checkpoint: epoch 38 `checkpoint_best.tar`
- Source checkpoint SHA-256:
  `49ab00661b8c4171712b72d120955e7928917e5fb94babf2545af888530d5fda`
- YOLOPointv11 was frozen during training.
- LightGlue was trained for the YOLOPointv11 128-dimensional descriptors.
- The LightGlue match threshold `0.1` is embedded in the exported graph.

## `yolopointv11_extractor.onnx`

SHA-256:
`8c2019acd5d0ed7732898b4f8e216894e80a090306588eefc528238fd4c3ecab`

This remains the lightweight feature-only export. Set
`yolopoint_object_detection_enable: false` to use it.

Input:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `image` | float32 | `[1, 3, H, W]` | RGB image normalized to `[0, 1]`; replicate a grayscale image into three channels |

`H` and `W` should be multiples of 16. The EuRoC input size `480 x 752` is
supported directly.

Outputs:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `semi` | float32 | `[1, 65, H/8, W/8]` | Keypoint logits; channel 64 is the dustbin |
| `descriptors` | float32 | `[1, 128, H/8, W/8]` | L2-normalized dense descriptor map |

## `yolopointv11_full.onnx`

The full export shares the same keypoint and descriptor outputs and also runs
the trained COCO object head. Set `yolopoint_object_detection_enable: true` to
select this model from YAML.

| Output | Type | Shape | Meaning |
|---|---|---|---|
| `semi` | float32 | `[1, 65, H/8, W/8]` | Keypoint cell logits |
| `descriptors` | float32 | `[1, 128, H/8, W/8]` | Dense descriptors |
| `detections` | float32 | `[1, N, 85]` | `xywh`, object confidence, 80 COCO class scores |

The C++ frontend multiplies object and class confidence, performs class-aware
NMS, and draws labeled boxes on `/vins_estimator/image_track`. The full model
uses the TensorRT cache under `trt_cache/full_extractor_dynamic`.

Relevant YAML controls:

```yaml
yolopoint_object_detection_enable: true
yolopoint_full_model_path: "onxx/yolopointv11_lightglue/yolopointv11_full.onnx"
yolopoint_full_trt_cache_path: "onxx/yolopointv11_lightglue/trt_cache/full_extractor_dynamic"
yolopoint_full_input_width: 768
yolopoint_full_input_height: 480
yolopoint_object_confidence_threshold: 0.25
yolopoint_object_iou_threshold: 0.45
yolopoint_object_max_detections: 100
```

Post-processing is intentionally kept outside the graph so score threshold,
NMS radius, candidate count, and spatial-uniformity policy remain configurable
in the VINS frontend. Decode `semi` with a channel-wise softmax, remove the
dustbin channel, apply an 8x pixel shuffle, then perform score thresholding and
NMS. Sample 128-dimensional descriptors from the dense descriptor map at the
retained keypoints using bilinear interpolation and L2-normalize them.

## `yolopointv11_lightglue.onnx`

SHA-256:
`06294ff91e762d550b7f5269a8f740212dbc2a26f66ef9c4c133c409adbbc29f`

Inputs:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `keypoints0` | float32 | `[1, N0, 2]` | Previous-left pixel coordinates `(x, y)` |
| `keypoints1` | float32 | `[1, N1, 2]` | Current-left pixel coordinates `(x, y)` |
| `descriptors0` | float32 | `[1, N0, 128]` | Previous-left L2-normalized descriptors |
| `descriptors1` | float32 | `[1, N1, 128]` | Current-left L2-normalized descriptors |
| `image_size0` | float32 | `[1, 2]` | Previous image size `(width, height)` |
| `image_size1` | float32 | `[1, 2]` | Current image size `(width, height)` |

`N0` and `N1` are independent dynamic dimensions, so the two frames do not
need equal keypoint counts and no padding features are required.

Outputs:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `matches0` | int64 | `[1, N0]` | Current-frame index for each previous point, or `-1` |
| `matching_scores0` | float32 | `[1, N0]` | Match confidence |

Only entries with `matches0 >= 0` are valid matches. The frontend should still
apply temporal geometric verification before inheriting feature IDs.

## Validation

Both files pass the ONNX checker and load in a clean ONNX Runtime process that
does not import `glue-factory`. CPU inference was tested with:

- extractor input `[1, 3, 480, 752]`;
- matcher inputs with different dynamic counts, `N0=23` and `N1=37`.

Comparison against the source PyTorch checkpoint produced identical match
indices. Maximum absolute differences were:

- extractor logits: `2.54e-5`;
- extractor descriptors: `1.05e-6`;
- LightGlue match scores: `3.79e-6`.

## TensorRT FP16

The original ONNX files remain FP32. TensorRT FP16 is enabled at runtime, so
CUDA fallback remains available for unsupported operators and no separate
FP16 ONNX model is required.

Cached engines:

| Model | Engine size | Optimization profile |
| --- | ---: | --- |
| extractor | 3.4 MB | `image=1x3x480x752` |
| LightGlue | 29.0 MB | independent `N0/N1`: min 1, opt 256, max 512 |

Engine SHA-256:

- extractor:
  `e044bf512cb86f41430bba7428a8716c4f07fabc3a24371ee75b1fadf5bf8c46`;
- LightGlue:
  `a7f48cc5f43f160a3f3628828180cc4cf81b9bc45751a34222ad2be6c8818051`.

The caches in `trt_cache/extractor` and
`trt_cache/lightglue_dynamic_n1_512_opt256` were built with TensorRT 10 for an
NVIDIA RTX 5070 Ti (`sm120`). TensorRT engine files are tied to the GPU
architecture, TensorRT version, model, and optimization profile. Rebuild them
rather than copying them to an incompatible deployment machine.

Use `lightglue_dynamic_n1_512_opt256` for the VINS frontend. Its two keypoint
axes are independent, so inputs such as `N0=50, N1=80` and `N0=512, N1=300`
are valid. Do not invoke LightGlue for an empty feature set; bypass matching
when either count is zero.

The ONNX Runtime TensorRT provider settings used to build and run these engines
were:

- FP16 enabled;
- engine cache enabled;
- 4 GiB maximum workspace;
- builder optimization level 3;
- TensorRT minimum subgraph size 1;
- CUDA and CPU providers retained as fallbacks.

An ONNX Runtime profile of one inference assigned each complete model graph to
one `TensorrtExecutionProvider` node. The TensorRT warning about the `matches0`
Int64 binding is benign for this export: the returned tensor was verified as
`int64` and its values were included in the numerical comparison below.

### FP16 numerical comparison

TensorRT FP16 was compared with ONNX Runtime CUDA FP32 on 12 consecutive real
left-camera frames from EuRoC `MH_01_easy`.

| Measurement | Result |
| --- | ---: |
| extractor logit relative L2 error | 0.3029% |
| descriptor relative L2 error | 0.3567% |
| dense descriptor mean cosine similarity | 0.9999937 |
| top-150 keypoint overlap, mean / minimum | 99.06% / 98.00% |

These measurements show only the inference precision difference; downstream
VINS trajectory accuracy must still be verified with a full dataset
evaluation.

### Dynamic LightGlue validation

The dynamic cache independently profiles both keypoint axes and their
corresponding descriptor axes:

```text
min: N0=1,   N1=1
opt: N0=256, N1=256
max: N0=512, N1=512
```

Functional tests covered `1/1`, `7/13`, `50/80`, `150/200`, `256/256`,
`300/512`, `512/300`, and `512/512`. All eight shapes executed successfully.
The runtime profile contained eight TensorRT node events and no CUDA or CPU
node fallback.

EuRoC validation used 11 consecutive frame pairs with independently varied
counts from `1/1` through `512/512`:

| Measurement | TensorRT FP16 result |
| --- | ---: |
| total match entries | 2649 |
| match tensor exact agreement | 99.7357% |
| valid matches, CUDA FP32 / TensorRT FP16 | 1676 / 1671 |
| common / union valid match pairs | 1670 / 1677 |
| valid match-pair Jaccard | 99.5826% |
| matching-score mean absolute error | 0.002823 |
| matching-score p99 absolute error | 0.068994 |

The real `512/512` frame pair produced identical match indices between CUDA
FP32 and TensorRT FP16.

### 1000-round latency benchmark

Environment: NVIDIA RTX 5070 Ti, extractor input `1x3x480x752`, 100 warm-up
rounds followed by 1000 measured rounds. Each measurement synchronizes CUDA,
so the reported time includes normal host-side ONNX Runtime invocation and
tensor transfers, but excludes keypoint post-processing, grid selection,
stereo KLT, and the VINS backend.

The extractor averaged `0.8767 ms` with TensorRT FP16, versus `1.4024 ms`
with the CUDA FP32 provider, a `1.60x` speedup. LightGlue results:

| `N0/N1` | Mean | Median | p95 | 1000-round total |
| --- | ---: | ---: | ---: | ---: |
| `150/150` | 0.8862 ms | 0.8744 ms | 1.0096 ms | 886.234 ms |
| `256/256` | 0.8473 ms | 0.8629 ms | 0.9313 ms | 847.253 ms |
| `512/300` | 1.2996 ms | 1.3070 ms | 1.4052 ms | 1299.610 ms |
| `512/512` | 1.4423 ms | 1.4459 ms | 1.6262 ms | 1442.340 ms |

## VINS `feature_tracker_type: 3`

The standalone models are integrated by
`vins/src/featureTracker/yolopoint_lightglue_feature_tracker.{h,cpp}`.
`config/euroc/euroc_stereo_imu_config.yaml` enables the new mode and contains
its complete parameter set.

The implemented frame path is:

1. Run YOLOPointv11 only on the current left grayscale image.
2. Decode, NMS, and retain up to 512 real candidates without padding.
3. Run dynamic LightGlue between retained previous-left observations and the
   current candidates, then apply fundamental-matrix geometric verification.
4. Inherit IDs for accepted temporal matches and assign new monotonically
   increasing IDs to selected unmatched points.
5. Prioritize tracked points, then response, while enforcing `MIN_DIST`; emit
   at most the VINS `max_cnt`.
6. Track the selected current-left points into the current-right image with
   CPU pyramidal LK, perform a right-to-left 0.5-pixel check, and apply
   disparity and epipolar constraints.
7. Compute left and right velocities from normalized camera coordinates and
   emit the existing VINS 7-element observation format.

For already rectified stereo input,
`yolopoint_lightglue_use_calibrated_epipolar: false` applies the direct
`|y_right-y_left|` pixel test. EuRoC topics contain raw camera images, so its
configuration enables calibration-aware epipolar error with the same
1.5-pixel-equivalent threshold.

The C++ integration was smoke-tested on 12 synchronized EuRoC `MH_01_easy`
stereo pairs:

- 512 YOLOPoint candidates per frame;
- 111--135 uniformly selected left observations per frame with
  `max_cnt: 150` and `min_dist: 30`;
- 833 temporal IDs inherited across the sequence;
- 244 right-camera KLT observations accepted;
- stable warmed-up total frontend latency about 9.5--10.2 ms per stereo pair
  on the RTX 5070 Ti, including CPU decode/NMS, descriptor sampling,
  fundamental-matrix verification, selection, and stereo KLT.

The TensorRT engine deserialization adds a substantial one-time startup cost
(about 36 seconds in the cached C++ smoke test) and is not part of per-frame
latency.
