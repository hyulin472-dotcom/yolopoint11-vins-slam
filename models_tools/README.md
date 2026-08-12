# Model deployment tools

The runtime ONNX models live only in `onxx/`. This directory contains
YAML-driven validation and local TensorRT generation tools; training weights
and generated artifacts are intentionally ignored by Git.

## One-command TensorRT generation

```bash
python3 -m pip install -r models_tools/requirements-runtime.txt
./tools/build_tensorrt_engines.sh --force
```

If TensorRT and ONNX Runtime GPU are provided by different Python environments:

```bash
TENSORRT_PYTHON=/path/to/tensorrt/python \
ONNXRUNTIME_PYTHON=/path/to/ort-gpu/python \
./tools/build_tensorrt_engines.sh --force
```

The script validates the ONNX graphs, builds standalone TensorRT engines, and
warms ONNX Runtime's TensorRT provider to create the exact caches consumed by
the C++ application. Outputs are placed below `onxx/*/trt_cache/` and must not
be committed. Engines are specific to GPU architecture, TensorRT/CUDA version,
model graph, precision, and optimization profile.

## Individual tools

- `validate_onnx.py`: validate graphs and configured tensor shapes.
- `build_tensorrt.py`: create standalone TensorRT `.engine` files.
- `build_ort_trt_cache.py`: create ONNX Runtime TensorRT engine/timing caches.
- `compare_models.py`: compare generated and reference model outputs.
- `export_onnx.py`: export ONNX when the original training repositories and
  checkpoints are available.
- `visualize_foundationstereo_kitti.py`: inspect metric KITTI depth output.

Configuration is in `models_tools/configs/`. Paths are resolved relative to
the project root; environment variables are supported for external training
repositories. The deployed ONNX files remain FP32, while TensorRT precision
can be FP32 or FP16.
