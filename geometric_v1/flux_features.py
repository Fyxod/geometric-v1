from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import DiffusionConfig
from .diffusion import _load_flux2_klein_pipe, resolve_device


@dataclass(frozen=True)
class CapturedFluxFeatures:
    feature: np.ndarray | None
    report: dict[str, Any]


def flux_transformer_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("enabled", False))


def _config_error(message: str, config: dict[str, Any]) -> CapturedFluxFeatures:
    return CapturedFluxFeatures(
        feature=None,
        report={
            "enabled": True,
            "available": False,
            "capture_method": str(config.get("capture_mode", "forward_hook")),
            "error": message,
            "warnings": [],
        },
    )


def _first_tensor(value: Any):
    try:
        import torch
    except Exception:
        return None

    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _pool_tensor(tensor: Any, pooling: str, max_elements: int) -> np.ndarray | None:
    try:
        import torch

        with torch.no_grad():
            value = tensor.detach()
            if not torch.is_floating_point(value):
                value = value.float()
            else:
                value = value.float()
            pooling = pooling.strip().lower()
            if pooling in {"cls", "first", "first_token", "token"} and value.ndim >= 3:
                pooled = value[:, 0, :].mean(dim=0)
            elif pooling in {"cls", "first", "first_token", "token"} and value.ndim >= 2:
                pooled = value[0]
            elif value.ndim <= 1:
                pooled = value.reshape(-1)
            elif value.ndim == 2:
                pooled = value.mean(dim=0)
            else:
                pooled = value.mean(dim=tuple(range(0, value.ndim - 1)))
            pooled = pooled.reshape(-1).detach().float().cpu()
            if pooled.numel() > max_elements:
                stride = max(1, math.ceil(pooled.numel() / max_elements))
                pooled = pooled[::stride][:max_elements]
            return pooled.numpy().astype(np.float32, copy=False)
    except Exception:
        return None


def _named_block_candidates(transformer: Any) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()

    preferred_attrs = (
        "transformer_blocks",
        "single_transformer_blocks",
        "joint_transformer_blocks",
        "double_transformer_blocks",
        "blocks",
        "layers",
    )
    for attr in preferred_attrs:
        blocks = getattr(transformer, attr, None)
        if blocks is None:
            continue
        if isinstance(blocks, dict):
            iterable = list(blocks.items())
        else:
            try:
                iterable = list(enumerate(blocks))
            except TypeError:
                iterable = []
        for index, module in iterable:
            if id(module) in seen or not hasattr(module, "register_forward_hook"):
                continue
            seen.add(id(module))
            candidates.append((f"{attr}.{index}", module))

    if candidates:
        return candidates

    named_modules = getattr(transformer, "named_modules", None)
    if not callable(named_modules):
        return candidates
    for name, module in named_modules():
        if not name or id(module) in seen or not hasattr(module, "register_forward_hook"):
            continue
        lower = name.lower()
        if "block" not in lower and "layer" not in lower:
            continue
        seen.add(id(module))
        candidates.append((name, module))
    return candidates


def _select_modules(transformer: Any, layers: list[Any]) -> tuple[list[tuple[str, Any]], list[str]]:
    warnings: list[str] = []
    candidates = _named_block_candidates(transformer)
    if not candidates:
        return [], ["No hookable Flux transformer blocks were found."]

    if not layers or layers == ["auto"] or "auto" in layers:
        if len(candidates) <= 3:
            return candidates, warnings
        indexes = sorted({0, len(candidates) // 2, len(candidates) - 1})
        selected = [candidates[index] for index in indexes]
        return selected, warnings

    selected: list[tuple[str, Any]] = []
    for layer in layers:
        matched: tuple[str, Any] | None = None
        if isinstance(layer, int) or (isinstance(layer, str) and layer.strip().lstrip("-").isdigit()):
            index = int(layer)
            if -len(candidates) <= index < len(candidates):
                matched = candidates[index]
        elif isinstance(layer, str):
            layer_name = layer.strip()
            for name, module in candidates:
                if name == layer_name or name.endswith(layer_name):
                    matched = (name, module)
                    break
        if matched is None:
            warnings.append(f"Requested Flux layer {layer!r} was not found.")
            continue
        if id(matched[1]) not in {id(module) for _, module in selected}:
            selected.append(matched)

    if not selected:
        warnings.append("No requested Flux layers matched; no features were captured.")
    return selected, warnings


def _bucket_for_call(index: int, total: int) -> str:
    if total <= 1:
        return "middle"
    fraction = index / max(total - 1, 1)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


class _FluxHookCapture:
    def __init__(self, modules: list[tuple[str, Any]], config: dict[str, Any], warnings: list[str]) -> None:
        self.modules = modules
        self.config = config
        self.warnings = list(warnings)
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def __enter__(self) -> "_FluxHookCapture":
        for name, module in self.modules:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                self.warnings.append(f"Layer {name} did not return a tensor-like output.")
                return
            max_elements = int(self.config.get("max_feature_elements", 262144))
            pooled = _pool_tensor(tensor, str(self.config.get("pooling", "mean")), max_elements)
            if pooled is None:
                self.warnings.append(f"Layer {name} output could not be pooled.")
                return
            self.records.append(
                {
                    "layer": name,
                    "call_index": len(self.records),
                    "feature_shape": list(tensor.shape),
                    "pooled_shape": list(pooled.shape),
                    "feature": pooled,
                }
            )

        return hook

    def captured(self) -> CapturedFluxFeatures:
        requested_timesteps = self.config.get("timesteps", ["early", "middle"])
        if not isinstance(requested_timesteps, list):
            requested_timesteps = [requested_timesteps]
        requested = {str(item).strip().lower() for item in requested_timesteps}
        include_all = not requested or "all" in requested or "*" in requested

        included: list[dict[str, Any]] = []
        total = len(self.records)
        for record in self.records:
            bucket = _bucket_for_call(int(record["call_index"]), total)
            record["timestep_bucket"] = bucket
            if include_all or bucket in requested:
                included.append(record)

        features = [record["feature"] for record in included if isinstance(record.get("feature"), np.ndarray)]
        feature = np.concatenate(features).astype(np.float32, copy=False) if features else None
        max_elements = int(self.config.get("max_feature_elements", 262144))
        if feature is not None and feature.size > max_elements:
            stride = max(1, math.ceil(feature.size / max_elements))
            feature = feature[::stride][:max_elements]

        warning_list = list(dict.fromkeys(self.warnings))
        if self.records:
            warning_list.append("Timestep labels are approximated from forward-hook call order.")

        return CapturedFluxFeatures(
            feature=feature,
            report={
                "enabled": True,
                "available": feature is not None and feature.size > 0,
                "capture_method": "forward_hook",
                "use_input_features": bool(self.config.get("use_input_features", False)),
                "use_denoising_features": bool(self.config.get("use_denoising_features", True)),
                "requested_layers": self.config.get("layers", ["auto"]),
                "layers_captured": sorted({record["layer"] for record in included}),
                "requested_timesteps": requested_timesteps,
                "timesteps_captured": sorted({record["timestep_bucket"] for record in included}),
                "feature_shapes": [
                    {"layer": record["layer"], "shape": record["feature_shape"], "timestep": record["timestep_bucket"]}
                    for record in included[:32]
                ],
                "pooled_feature_shapes": [
                    {"layer": record["layer"], "shape": record["pooled_shape"], "timestep": record["timestep_bucket"]}
                    for record in included[:32]
                ],
                "captured_call_count": total,
                "included_feature_count": len(included),
                "warnings": warning_list,
            },
        )


def _build_capture(diffusion: DiffusionConfig, config: dict[str, Any]) -> _FluxHookCapture:
    if diffusion.selected_model != "flux2_klein":
        raise RuntimeError("Flux transformer feature loss requires diffusion.selected_model='flux2_klein'.")
    if str(config.get("capture_mode", "forward_hook")) != "forward_hook":
        raise RuntimeError("Only capture_mode='forward_hook' is currently implemented.")
    device = resolve_device(diffusion)
    flux = diffusion.flux2_klein
    pipe = _load_flux2_klein_pipe(flux.model_id, device, flux.torch_dtype, flux.cpu_offload)
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise RuntimeError("Active Flux pipeline does not expose a transformer attribute.")
    layers = config.get("layers", ["auto"])
    if not isinstance(layers, list):
        layers = [layers]
    modules, warnings = _select_modules(transformer, layers)
    if not modules:
        raise RuntimeError("; ".join(warnings) or "No Flux transformer layers could be selected.")
    return _FluxHookCapture(modules, config, warnings)


def capture_flux_features_during_generation(
    diffusion: DiffusionConfig,
    config: dict[str, Any] | None,
    generate: Callable[[], Any],
) -> tuple[Any, CapturedFluxFeatures]:
    if not flux_transformer_enabled(config):
        image = generate()
        return image, CapturedFluxFeatures(None, {"enabled": False, "available": False})

    assert config is not None
    strict = bool(config.get("strict", False))
    if not bool(config.get("use_denoising_features", True)):
        message = (
            "Stable pre-generation Flux input feature capture is not implemented for this "
            "Diffusers pipeline; enable use_denoising_features to use forward hooks."
        )
        if strict:
            raise RuntimeError(message)
        image = generate()
        report = _config_error(message, config)
        report.report["use_input_features"] = bool(config.get("use_input_features", False))
        report.report["use_denoising_features"] = False
        return image, report

    try:
        capture = _build_capture(diffusion, config)
    except Exception as exc:
        if strict:
            raise
        image = generate()
        return image, _config_error(f"{type(exc).__name__}: {exc}", config)

    try:
        import torch

        with capture:
            with torch.inference_mode():
                image = generate()
    except Exception:
        raise

    captured = capture.captured()
    if strict and not captured.report.get("available"):
        raise RuntimeError("Flux transformer features were enabled with strict=true but no features were captured.")
    return image, captured


def compare_flux_features(
    original: CapturedFluxFeatures | None,
    candidate: CapturedFluxFeatures | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not flux_transformer_enabled(config):
        return {"enabled": False, "available": False}
    assert config is not None
    metric: dict[str, Any] = {
        "enabled": True,
        "available": False,
        "capture_method": str(config.get("capture_mode", "forward_hook")),
        "distance_type": str(config.get("distance", "cosine")),
        "weight": float(config.get("weight", 1.0)),
        "use_input_features": bool(config.get("use_input_features", False)),
        "use_denoising_features": bool(config.get("use_denoising_features", True)),
        "original": original.report if original is not None else {"available": False},
        "candidate": candidate.report if candidate is not None else {"available": False},
    }
    metric["warnings"] = list(
        dict.fromkeys(
            list(metric["original"].get("warnings", []))
            + list(metric["candidate"].get("warnings", []))
        )
    )
    if original is None or candidate is None or original.feature is None or candidate.feature is None:
        metric["error"] = "Original or candidate Flux features are unavailable."
        return metric

    a = np.asarray(original.feature, dtype=np.float64).reshape(-1)
    b = np.asarray(candidate.feature, dtype=np.float64).reshape(-1)
    size = min(a.size, b.size)
    if size <= 0:
        metric["error"] = "Flux feature vectors are empty."
        return metric
    a = a[:size]
    b = b[:size]
    distance_type = str(config.get("distance", "cosine")).strip().lower()
    if distance_type in {"normalized_l2", "l2"}:
        distance = float(np.linalg.norm(a - b) / max(math.sqrt(size), 1.0))
    else:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        distance = 1.0 if denom <= 1e-12 else float(1.0 - np.dot(a, b) / denom)
        distance_type = "cosine"
    metric.update(
        {
            "available": True,
            "distance": distance,
            "distance_type": distance_type,
            "layers_captured": metric["candidate"].get("layers_captured", []),
            "timesteps_captured": metric["candidate"].get("timesteps_captured", []),
        }
    )
    return metric
