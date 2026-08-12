#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-python3}"
trt_python="${TENSORRT_PYTHON:-${python_bin}}"
ort_python="${ONNXRUNTIME_PYTHON:-${python_bin}}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: NVIDIA driver tools were not found" >&2
  exit 1
fi
if ! "${trt_python}" -c 'import tensorrt, onnx' >/dev/null 2>&1; then
  echo "error: ${trt_python} must provide tensorrt and onnx" >&2
  exit 1
fi
if ! "${ort_python}" -c 'import onnxruntime as o; assert "TensorrtExecutionProvider" in o.get_available_providers()' >/dev/null 2>&1; then
  echo "error: ${ort_python} does not provide ONNX Runtime's TensorRT provider" >&2
  echo "hint: install models_tools/requirements-runtime.txt in a GPU-enabled environment" >&2
  exit 1
fi

cd "${project_root}"
configs=(
  models_tools/configs/yolopoint_lightglue.yaml
  models_tools/configs/fast_foundationstereo.yaml
)
for config in "${configs[@]}"; do
  "${trt_python}" models_tools/scripts/validate_onnx.py --config "${config}"
  "${trt_python}" models_tools/scripts/build_tensorrt.py --config "${config}" "$@"
  "${ort_python}" models_tools/scripts/build_ort_trt_cache.py --config "${config}"
done

echo "TensorRT artifacts were generated under ${project_root}/onxx/*/trt_cache"
