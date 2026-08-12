#!/usr/bin/env python3
"""Export deployment ONNX models from YAML configuration files."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ensure_parent, load_config, resolve_path, select_components


class YOLOPointExtractorWrapper(nn.Module):
    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Export only the shared encoder and local-feature heads. Calling the
        # complete YOLOPoint model would also execute the unused object head,
        # whose multi-scale concatenations require dimensions divisible by 32.
        # The local heads themselves support the VINS requirement (H/W / 8),
        # including EuRoC's 480x752 input.
        model = self.extractor.model.model
        encoded = model.Conv1(image)
        encoded = model.Conv2(encoded)
        skip = model.Bottleneck1(encoded)
        encoded = model.Conv3(skip)
        semi = model.BottleneckDet(encoded)
        encoded = model.Bottleneck2(encoded)
        desc_a = model.MaxPool(skip)
        desc_b = model.ups(model.ConvDescB(encoded))
        descriptors = model.BottleneckDesc(torch.cat((desc_a, desc_b), dim=1))
        descriptors = F.normalize(descriptors, p=2, dim=1)
        return semi, descriptors


class LightGlueWrapper(nn.Module):
    def __init__(self, matcher: nn.Module):
        super().__init__()
        self.matcher = matcher

    def forward(
        self,
        keypoints0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
        image_size0: torch.Tensor,
        image_size1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # This is the inference-only LightGlue path used by VINS.  Building the
        # padded training assignment matrix with in-place slice writes makes
        # the legacy ONNX exporter emit an invalid dynamic-rank transpose in
        # recent PyTorch versions.  The padded dustbin row/column is not used
        # by filter_matches, so calculate its mathematically identical inner
        # score matrix directly.
        def normalize_keypoints(kpts: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
            shift = size.to(kpts) / 2
            scale = size.to(kpts).max(-1).values / 2
            return (kpts - shift[..., None, :]) / scale[..., None, None]

        kpts0 = normalize_keypoints(keypoints0, image_size0)
        kpts1 = normalize_keypoints(keypoints1, image_size1)
        desc0 = self.matcher.input_proj(descriptors0.contiguous())
        desc1 = self.matcher.input_proj(descriptors1.contiguous())
        encoding0 = self.matcher.posenc(kpts0)
        encoding1 = self.matcher.posenc(kpts1)
        for transformer in self.matcher.transformers:
            desc0, desc1 = transformer(desc0, desc1, encoding0, encoding1)

        assignment = self.matcher.log_assignment[-1]
        mdesc0 = assignment.final_proj(desc0)
        mdesc1 = assignment.final_proj(desc1)
        descriptor_dim = float(mdesc0.shape[-1])
        mdesc0 = mdesc0 / descriptor_dim**0.25
        mdesc1 = mdesc1 / descriptor_dim**0.25
        similarity = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = assignment.matchability(desc0)
        z1 = assignment.matchability(desc1)
        certainty = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        scores = (
            F.log_softmax(similarity, dim=2)
            + F.log_softmax(similarity.transpose(-1, -2).contiguous(), dim=2).transpose(-1, -2)
            + certainty
        )

        max0 = scores.max(2)
        max1 = scores.max(1)
        matches0 = max0.indices
        indices0 = torch.arange(matches0.shape[1], device=matches0.device)[None]
        mutual0 = indices0 == max1.indices.gather(1, matches0)
        match_scores0 = torch.where(mutual0, max0.values.exp(), max0.values.new_tensor(0))
        valid0 = mutual0 & (match_scores0 > float(self.matcher.conf.filter_threshold))
        matches0 = torch.where(valid0, matches0, -1)
        return matches0, match_scores0


def _load_yolopoint_pipeline(config: Mapping[str, Any]) -> nn.Module:
    source = config["source"]
    source_root = resolve_path(config, source["root"], required=True)
    checkpoint = resolve_path(config, source["checkpoint"], required=True)
    extractor_weights = resolve_path(config, source["extractor_weights"], required=True)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from gluefactory.models.two_view_pipeline import TwoViewPipeline

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_conf = copy.deepcopy(saved["conf"]["model"])
    model_conf["extractor"]["weights"] = str(extractor_weights)
    pipeline = TwoViewPipeline(model_conf)
    result = pipeline.load_state_dict(saved["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return pipeline.eval()


def _export_yolopoint_lightglue(config: Mapping[str, Any], names: list[str], force: bool) -> None:
    pipeline = _load_yolopoint_pipeline(config)
    opset_default = int(config.get("onnx", {}).get("opset", 17))

    if "extractor" in names:
        component = config["components"]["extractor"]
        output = resolve_path(config, component["onnx_path"])
        if output.exists() and not force:
            print(f"skip existing ONNX: {output}")
        else:
            ensure_parent(output)
            height, width = map(int, component["input_size"])
            dummy = torch.randn(1, 3, height, width)
            dynamic_axes = None
            if component.get("dynamic_spatial", False):
                dynamic_axes = {
                    "image": {2: "height", 3: "width"},
                    "semi": {2: "height_div8", 3: "width_div8"},
                    "descriptors": {2: "height_div8", 3: "width_div8"},
                }
            torch.onnx.export(
                YOLOPointExtractorWrapper(pipeline.extractor),
                dummy,
                str(output),
                input_names=["image"],
                output_names=["semi", "descriptors"],
                dynamic_axes=dynamic_axes,
                opset_version=int(component.get("opset", opset_default)),
                do_constant_folding=True,
                dynamo=False,
            )
            print(f"exported extractor: {output}")

    if "lightglue" in names:
        component = config["components"]["lightglue"]
        output = resolve_path(config, component["onnx_path"])
        if output.exists() and not force:
            print(f"skip existing ONNX: {output}")
        else:
            ensure_parent(output)
            count = int(component.get("example_keypoints", 256))
            descriptor_dim = int(component.get("descriptor_dim", 128))
            height, width = map(float, component["image_size"])
            args = (
                torch.rand(1, count, 2) * torch.tensor([width, height]),
                torch.rand(1, count, 2) * torch.tensor([width, height]),
                F.normalize(torch.rand(1, count, descriptor_dim), p=2, dim=-1),
                F.normalize(torch.rand(1, count, descriptor_dim), p=2, dim=-1),
                torch.tensor([[width, height]], dtype=torch.float32),
                torch.tensor([[width, height]], dtype=torch.float32),
            )
            dynamic_axes = None
            if component.get("dynamic", True):
                dynamic_axes = {
                    "keypoints0": {1: "num_keypoints0"},
                    "keypoints1": {1: "num_keypoints1"},
                    "descriptors0": {1: "num_keypoints0"},
                    "descriptors1": {1: "num_keypoints1"},
                    "matches0": {1: "num_keypoints0"},
                    "matching_scores0": {1: "num_keypoints0"},
                }
            torch.onnx.export(
                LightGlueWrapper(pipeline.matcher),
                args,
                str(output),
                input_names=[
                    "keypoints0", "keypoints1", "descriptors0", "descriptors1",
                    "image_size0", "image_size1",
                ],
                output_names=["matches0", "matching_scores0"],
                dynamic_axes=dynamic_axes,
                opset_version=int(component.get("opset", 18)),
                do_constant_folding=True,
                # The dynamo exporter handles negative transpose axes in the
                # dynamic LightGlue attention graph correctly on current
                # PyTorch releases; the legacy JIT exporter does not.
                dynamo=True,
                external_data=False,
            )
            stale_sidecar = Path(str(output) + ".data")
            if stale_sidecar.exists():
                stale_sidecar.unlink()
            print(f"exported LightGlue: {output}")


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("foundation_export_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_foundation_stereo(config: Mapping[str, Any], names: list[str], force: bool) -> None:
    if names != ["stereo"]:
        raise ValueError("Fast-FoundationStereo configuration only supports component 'stereo'")
    source = config["source"]
    source_root = resolve_path(config, source["root"], required=True)
    checkpoint = resolve_path(config, source["checkpoint"], required=True)
    exporter_path = source_root / source.get("export_module", "scripts/make_single_onnx.py")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    module = _load_module(exporter_path)

    component = config["components"]["stereo"]
    output = resolve_path(config, component["onnx_path"])
    if output.exists() and not force:
        print(f"skip existing ONNX: {output}")
        return
    ensure_parent(output)
    height, width = map(int, component["input_size"])
    if height % 32 or width % 32:
        raise ValueError("Fast-FoundationStereo input height and width must be divisible by 32")
    device_name = str(config.get("export", {}).get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is not available")
    device = torch.device(device_name)

    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.args.max_disp = int(component.get("max_disp", 192))
    model.args.valid_iters = int(component.get("valid_iters", 4))
    model.args.mixed_precision = False
    model.to(device).eval()
    wrapper = module.FastFoundationStereoSingleOnnx(model).to(device).eval()
    module._fs_module.normalize_image = lambda image: image
    module._fs_module.build_gwc_volume_optimized_pytorch1 = module._build_gwc_volume_onnx
    module._fs_module.build_concat_volume_optimized_pytorch1 = module._build_concat_volume_onnx
    left = torch.randn(1, 3, height, width, device=device)
    right = torch.randn(1, 3, height, width, device=device)
    torch.onnx.export(
        wrapper,
        (left, right),
        str(output),
        input_names=["left_image", "right_image"],
        output_names=["disparity"],
        opset_version=int(component.get("opset", config.get("onnx", {}).get("opset", 17))),
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported Fast-FoundationStereo: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--component", default="all", help="component name, comma list, or all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    names = select_components(config, args.component)
    exporter = config.get("exporter")
    if exporter == "yolopoint_lightglue":
        _export_yolopoint_lightglue(config, names, args.force)
    elif exporter == "fast_foundationstereo":
        _export_foundation_stereo(config, names, args.force)
    else:
        raise ValueError(f"unsupported exporter: {exporter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
