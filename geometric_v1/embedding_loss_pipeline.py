from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_DEEPFACE_MODELS,
    DeepFaceConfig,
    DiffusionConfig,
    PerturbationStep,
    _deepface_from_dict,
    _diffusion_from_dict,
    _perturbation_from_dict,
    perturbation_to_report,
)
from .deepface_compare import compare_images
from .diffusion import (
    _flux_dimensions,
    _load_flux2_klein_pipe,
    _load_instruct_pix2pix_pipe,
    _resize_for_diffusion,
    edit_image,
    resolve_device,
    selected_diffusion_model,
)
from .image_io import load_image, load_pil_image, save_image, save_pil
from .loss_pipeline import (
    _config_block,
    _deep_update,
    _enabled,
    _force_diffusion_seed,
    _initial_vector,
    _lpips_distance,
    _metric_config,
    _parameters_from_vector,
    _parse_bounds,
    _psnr,
    _read_json,
    _resolve,
    _selected_loss_seed,
    _ssim,
    _steps_from_parameters,
    _write_json,
)
from .perturbations import apply_perturbation_pipeline


_CLIP_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


@dataclass(frozen=True)
class EmbeddingLossConfig:
    config_path: Path
    raw: dict[str, Any]
    pipeline_config: Path | None
    input_path: Path
    prompt: str
    output_parent: Path
    seed: int
    configured_seed: int
    randomize_seed: bool
    random_seed_range: tuple[int, int]
    perturbation_templates: list[PerturbationStep]
    diffusion: DiffusionConfig
    identity: DeepFaceConfig
    optimizer: dict[str, Any]
    objective: dict[str, Any]
    initialization: str
    initial_values: dict[str, float]
    parameter_specs: list[Any]


def _identity_config_block(data: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    block = data.get("identity")
    reference_block = reference.get("deepface", {})
    if block is None:
        return deepcopy(reference_block) if isinstance(reference_block, dict) else {}
    if not isinstance(block, dict):
        raise ValueError("identity must be an object")
    if block.get("from_pipeline", False):
        overrides = block.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("identity.overrides must be an object")
        return _deep_update(reference_block if isinstance(reference_block, dict) else {}, overrides)
    return block


def load_embedding_loss_config(path: Path) -> EmbeddingLossConfig:
    path = path.resolve()
    data = _read_json(path)
    base_dir = path.parent
    pipeline_config = data.get("pipeline_config")
    pipeline_path = _resolve(base_dir, str(pipeline_config)).resolve() if pipeline_config else None
    reference = _read_json(pipeline_path) if pipeline_path else {}

    configured_seed, seed, randomize_seed, seed_range = _selected_loss_seed(data, reference)
    data = deepcopy(data)
    data["configured_seed"] = configured_seed
    data["seed"] = seed
    data["randomize_seed"] = randomize_seed
    data["random_seed_range"] = list(seed_range)

    input_value = data.get("input", reference.get("input"))
    if input_value is None:
        raise ValueError("embedding loss config must include input or pipeline_config with input")
    prompt = str(data.get("prompt", reference.get("prompt", "")))
    if not prompt:
        raise ValueError("embedding loss config must include prompt or pipeline_config with prompt")

    perturbation_items = data.get("perturbations", reference.get("perturbations", []))
    if not isinstance(perturbation_items, list) or not perturbation_items:
        raise ValueError("embedding loss config must include at least one perturbation template")
    perturbations = [
        _perturbation_from_dict(item, seed, index)
        for index, item in enumerate(perturbation_items)
    ]

    diffusion_values = _force_diffusion_seed(_config_block(data, reference, "diffusion"), seed)
    identity_values = _identity_config_block(data, reference)

    parameter_config = data.get("parameters", {})
    if not isinstance(parameter_config, dict):
        raise ValueError("parameters must be an object")
    bounds = parameter_config.get("bounds", {})
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("parameters.bounds must include at least one optimizable parameter")
    initial_values = parameter_config.get("initial_values", {})
    if not isinstance(initial_values, dict):
        raise ValueError("parameters.initial_values must be an object")

    return EmbeddingLossConfig(
        config_path=path,
        raw=data,
        pipeline_config=pipeline_path,
        input_path=_resolve(base_dir, str(input_value)).resolve(),
        prompt=prompt,
        output_parent=_resolve(base_dir, str(data.get("output_dir", "output"))).resolve(),
        seed=seed,
        configured_seed=configured_seed,
        randomize_seed=randomize_seed,
        random_seed_range=seed_range,
        perturbation_templates=perturbations,
        diffusion=_diffusion_from_dict(diffusion_values, seed),
        identity=_deepface_from_dict(identity_values),
        optimizer=data.get("optimizer", {}) if isinstance(data.get("optimizer", {}), dict) else {},
        objective=data.get("objective", {}) if isinstance(data.get("objective", {}), dict) else {},
        initialization=str(parameter_config.get("initialization", "fixed")),
        initial_values={str(key): float(value) for key, value in initial_values.items() if not isinstance(value, dict)},
        parameter_specs=_parse_bounds(bounds),
    )


def _weight(config: dict[str, Any], default: float) -> float:
    return float(config.get("weight", default))


def _objective_models(config: EmbeddingLossConfig) -> dict[str, bool]:
    identity_objective = _metric_config(config.objective, "identity")
    models = dict(config.identity.models)
    objective_models = identity_objective.get("models")
    if isinstance(objective_models, dict):
        for model_name in DEFAULT_DEEPFACE_MODELS:
            if model_name in objective_models:
                models[model_name] = bool(objective_models[model_name])
    return {
        model_name: bool(models.get(model_name, False))
        for model_name in DEFAULT_DEEPFACE_MODELS
    }


def _strict_identity(config: EmbeddingLossConfig) -> bool:
    return bool(_metric_config(config.objective, "identity").get("strict", False))


def _vector_distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        size = min(a.size, b.size)
        a = a[:size]
        b = b[:size]
    diff = a - b
    l2 = float(np.linalg.norm(diff) / max(math.sqrt(max(a.size, 1)), 1.0))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine_distance = 1.0 if denom <= 1e-12 else float(1.0 - np.dot(a, b) / denom)
    return {"l2": l2, "cosine_distance": cosine_distance}


def _image_l2(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def _extract_identity_embedding(image_path: Path, model_name: str, config: DeepFaceConfig) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=str(image_path),
            model_name=model_name,
            detector_backend=config.detector_backend,
            enforce_detection=config.enforce_detection,
            align=config.align,
        )
        if isinstance(result, list) and result:
            first = result[0]
            embedding = first.get("embedding") if isinstance(first, dict) else first
        elif isinstance(result, dict):
            embedding = result.get("embedding")
        else:
            embedding = result
        if embedding is None:
            raise RuntimeError("DeepFace.represent did not return an embedding")
        return np.asarray(embedding, dtype=np.float64), {"ok": True, "source": "direct_embedding"}
    except Exception as exc:
        return None, {"ok": False, "source": "direct_embedding", "error": f"{type(exc).__name__}: {exc}"}


def _verify_identity_pair(image_a: Path, image_b: Path, model_name: str, config: DeepFaceConfig) -> dict[str, Any]:
    report = compare_images(
        image_a,
        image_b,
        config,
        models={model_name: True},
        allow_parallel=False,
        event_context={"stage": "embedding_loss_identity_fallback"},
    )
    result = report.get("models", {}).get(model_name, {})
    if result.get("ok") and result.get("match_percent") is not None:
        similarity = float(result["match_percent"])
        return {
            "ok": True,
            "source": "verify_distance",
            "similarity_percent": similarity,
            "distance_percent": max(0.0, 100.0 - similarity),
            "raw_distance": result.get("distance"),
            "threshold": result.get("threshold"),
            "verified": result.get("verified"),
        }
    return {
        "ok": False,
        "source": "verify_distance",
        "error": result.get("error", "DeepFace.verify failed"),
    }


def _identity_pair_metrics(
    image_a: Path,
    image_b: Path,
    config: DeepFaceConfig,
    enabled_models: dict[str, bool],
    cached_a: dict[str, dict[str, Any]],
    strict: bool,
) -> dict[str, Any]:
    per_model: dict[str, Any] = {}
    similarities: list[float] = []
    distances: list[float] = []

    for model_name, enabled in enabled_models.items():
        if not enabled:
            per_model[model_name] = {"enabled": False, "skipped": True}
            continue
        cached = cached_a.get(model_name, {})
        cached_embedding = cached.get("embedding")
        embedding_b, info_b = _extract_identity_embedding(image_b, model_name, config)
        if isinstance(cached_embedding, np.ndarray) and embedding_b is not None:
            distances_raw = _vector_distance(cached_embedding, embedding_b)
            cosine = distances_raw["cosine_distance"]
            similarity = max(0.0, min(100.0, 100.0 * (1.0 - cosine / 2.0)))
            distance_percent = max(0.0, 100.0 - similarity)
            metric = {
                "enabled": True,
                "ok": True,
                "source": "direct_embedding",
                "similarity_percent": similarity,
                "distance_percent": distance_percent,
                "raw_distance": cosine,
                "threshold": None,
                "embedding_l2": distances_raw["l2"],
            }
        else:
            metric = _verify_identity_pair(image_a, image_b, model_name, config)
            metric["enabled"] = True
            metric["direct_embedding_a"] = cached.get("info", {})
            metric["direct_embedding_b"] = info_b

        if metric.get("ok"):
            similarities.append(float(metric["similarity_percent"]))
            distances.append(float(metric["distance_percent"]))
        elif strict:
            raise RuntimeError(f"identity metric failed for {model_name}: {metric.get('error')}")
        per_model[model_name] = metric

    return {
        "models": per_model,
        "mean_similarity_percent": sum(similarities) / len(similarities) if similarities else None,
        "mean_distance_percent": sum(distances) / len(distances) if distances else None,
        "ok_model_count": len(similarities),
    }


def _cache_identity_embeddings(image_path: Path, config: DeepFaceConfig, enabled_models: dict[str, bool], strict: bool) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for model_name, enabled in enabled_models.items():
        if not enabled:
            cache[model_name] = {"enabled": False, "skipped": True}
            continue
        embedding, info = _extract_identity_embedding(image_path, model_name, config)
        if embedding is None and strict:
            raise RuntimeError(f"identity embedding cache failed for {model_name}: {info.get('error')}")
        cache[model_name] = {"enabled": True, "embedding": embedding, "info": info}
    return cache


def _pil_to_vae_tensor(image_path: Path, diffusion: DiffusionConfig):
    import torch

    image = load_pil_image(image_path)
    if diffusion.selected_model == "flux2_klein":
        height, width = _flux_dimensions(image, diffusion)
        image = image.convert("RGB").resize((width, height))
    else:
        image = _resize_for_diffusion(image, diffusion.max_size)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    device = resolve_device(diffusion)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _active_pipe(diffusion: DiffusionConfig):
    device = resolve_device(diffusion)
    if diffusion.selected_model == "flux2_klein":
        flux = diffusion.flux2_klein
        return _load_flux2_klein_pipe(flux.model_id, device, flux.torch_dtype, flux.cpu_offload)
    return _load_instruct_pix2pix_pipe(diffusion.model_id, device)


def _vae_latent(image_path: Path, diffusion: DiffusionConfig) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        import torch

        pipe = _active_pipe(diffusion)
        vae = getattr(pipe, "vae", None)
        if vae is None or not hasattr(vae, "encode"):
            return None, {"ok": False, "error": "active diffusion pipeline does not expose a VAE encoder"}
        tensor = _pil_to_vae_tensor(image_path, diffusion)
        dtype = next(vae.parameters()).dtype if hasattr(vae, "parameters") else tensor.dtype
        tensor = tensor.to(dtype=dtype)
        with torch.no_grad():
            encoded = vae.encode(tensor)
            if hasattr(encoded, "latent_dist"):
                latent = encoded.latent_dist.mean
            elif hasattr(encoded, "latents"):
                latent = encoded.latents
            else:
                latent = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
        return latent.detach().float().cpu().numpy(), {
            "ok": True,
            "source": "diffusion_pipeline_vae",
            "device": resolve_device(diffusion),
            "selected_model": diffusion.selected_model,
        }
    except Exception as exc:
        return None, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _get_clip_model(model_id: str, device: str):
    key = (model_id, device)
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    _CLIP_CACHE[key] = (model, processor)
    return model, processor


def _clip_embedding(image_path: Path, model_id: str, diffusion: DiffusionConfig) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        import torch

        device = resolve_device(diffusion)
        model, processor = _get_clip_model(model_id, device)
        image = load_pil_image(image_path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().float().cpu().numpy(), {"ok": True, "model_id": model_id, "device": device}
    except Exception as exc:
        return None, {"ok": False, "model_id": model_id, "error": f"{type(exc).__name__}: {exc}"}


def _pair_distance(cache_value: np.ndarray | None, candidate_value: np.ndarray | None, strict: bool, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if cache_value is None or candidate_value is None:
        metric = {"ok": False, "error": f"{label} unavailable"}
        if strict:
            raise RuntimeError(metric["error"])
        return {}, metric
    return _vector_distance(cache_value, candidate_value), {"ok": True}


def _input_stealth_metrics(
    original: np.ndarray,
    perturbed: np.ndarray,
    original_path: Path,
    perturbed_path: Path,
    objective: dict[str, Any],
) -> dict[str, Any]:
    config = _metric_config(objective, "input_stealth")
    if not _enabled(config, True):
        return {}
    metrics: dict[str, Any] = {}
    if bool(config.get("use_psnr", True)):
        metrics["psnr"] = _psnr(original, perturbed)
    if bool(config.get("use_ssim", True)):
        metrics["ssim"] = _ssim(original, perturbed)
    if bool(config.get("use_lpips", False)):
        value, error = _lpips_distance(original_path, perturbed_path)
        metrics["lpips"] = value
        if error:
            metrics["lpips_error"] = error
    return metrics


def _output_disruption_metrics(
    original_diffused: np.ndarray,
    perturbed_diffused: np.ndarray,
    original_diffused_path: Path,
    perturbed_diffused_path: Path,
    objective: dict[str, Any],
) -> dict[str, Any]:
    config = _metric_config(objective, "output_disruption")
    if not _enabled(config, True):
        return {}
    metrics: dict[str, Any] = {}
    if bool(config.get("use_pixel_l2", True)):
        metrics["pixel_l2"] = _image_l2(original_diffused, perturbed_diffused)
    if bool(config.get("use_ssim_drop", True)):
        ssim = _ssim(original_diffused, perturbed_diffused)
        metrics["ssim"] = ssim
        metrics["ssim_drop"] = max(0.0, 1.0 - ssim)
    if bool(config.get("use_lpips", False)):
        value, error = _lpips_distance(original_diffused_path, perturbed_diffused_path)
        metrics["lpips"] = value
        if error:
            metrics["lpips_error"] = error
    return metrics


def _loss_from_embedding_metrics(
    objective: dict[str, Any],
    metrics: dict[str, Any],
    vector: np.ndarray,
    initial_vector: np.ndarray,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    total = 0.0

    stealth_config = _metric_config(objective, "input_stealth")
    stealth = metrics.get("input_stealth", {})
    if _enabled(stealth_config, True):
        for name, default_target, direction, default_weight in (
            ("psnr", 28.0, "min", 2.0),
            ("ssim", 0.88, "min", 25.0),
            ("lpips", 0.25, "max", 2.0),
        ):
            if not bool(stealth_config.get(f"use_{name}", name in {"psnr", "ssim"})):
                continue
            metric_config = _metric_config(objective, "input_stealth", name)
            value = stealth.get(name)
            if value is None:
                continue
            target = float(metric_config.get("target", default_target))
            if direction == "min":
                violation = max(0.0, target - float(value))
            else:
                violation = max(0.0, float(value) - target)
            components[f"input_{name}_penalty"] = _weight(metric_config, default_weight) * violation * violation

    identity_config = _metric_config(objective, "identity")
    identity = metrics.get("identity", {})
    if _enabled(identity_config, True):
        if bool(identity_config.get("use_pre_identity", True)):
            pre = identity.get("pre", {}).get("mean_similarity_percent")
            if pre is not None:
                target = float(identity_config.get("pre_identity_target", 90.0))
                violation = max(0.0, target - float(pre))
                components["pre_identity_penalty"] = float(identity_config.get("pre_identity_weight", 5.0)) * violation * violation
        if bool(identity_config.get("use_post_identity", True)):
            post_distance = identity.get("post", {}).get("mean_distance_percent")
            if post_distance is not None:
                components["post_identity_reward"] = -float(identity_config.get("post_identity_distance_weight", 1.0)) * float(post_distance)

    vae_config = _metric_config(objective, "vae_latent")
    vae = metrics.get("vae_latent", {})
    if _enabled(vae_config, True):
        if bool(vae_config.get("use_input_vae", False)) and isinstance(vae.get("input"), dict):
            components["input_vae_distance_penalty"] = float(vae_config.get("input_distance_weight", 0.0)) * float(vae["input"].get("l2", 0.0))
        if bool(vae_config.get("use_output_vae", True)) and isinstance(vae.get("output"), dict):
            components["output_vae_distance_reward"] = -float(vae_config.get("output_distance_weight", 1.0)) * float(vae["output"].get("l2", 0.0))

    clip_config = _metric_config(objective, "clip_image")
    clip = metrics.get("clip_image", {})
    if _enabled(clip_config, False):
        if bool(clip_config.get("use_input_clip", False)) and isinstance(clip.get("input"), dict):
            components["input_clip_distance_penalty"] = float(clip_config.get("input_distance_weight", 0.0)) * float(clip["input"].get("cosine_distance", 0.0))
        if bool(clip_config.get("use_output_clip", True)) and isinstance(clip.get("output"), dict):
            components["output_clip_distance_reward"] = -float(clip_config.get("output_distance_weight", 1.0)) * float(clip["output"].get("cosine_distance", 0.0))

    disruption_config = _metric_config(objective, "output_disruption")
    disruption = metrics.get("output_disruption", {})
    if _enabled(disruption_config, True):
        if bool(disruption_config.get("use_pixel_l2", True)) and disruption.get("pixel_l2") is not None:
            components["output_pixel_l2_reward"] = -float(disruption_config.get("pixel_l2_distance_weight", 1.0)) * float(disruption["pixel_l2"])
        if bool(disruption_config.get("use_ssim_drop", True)) and disruption.get("ssim_drop") is not None:
            components["output_ssim_drop_reward"] = -float(disruption_config.get("ssim_drop_weight", 1.0)) * float(disruption["ssim_drop"])
        if bool(disruption_config.get("use_lpips", False)) and disruption.get("lpips") is not None:
            components["output_lpips_reward"] = -float(disruption_config.get("lpips_distance_weight", 1.0)) * float(disruption["lpips"])

    reg_config = _metric_config(objective, "parameter_regularization")
    if _enabled(reg_config, True):
        diff = vector - initial_vector
        components["parameter_regularization_penalty"] = _weight(reg_config, 0.01) * float(diff @ diff)

    for value in components.values():
        total += float(value)
    return total, components


class EmbeddingLossRunner:
    def __init__(self, config: EmbeddingLossConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.output_parent / f"embedding_loss_run_{timestamp}"
        self.iterations_dir = self.run_dir / "iterations"
        self.best_dir = self.run_dir / "best"
        self.original_path = self.run_dir / "original.png"
        self.original_diffused_path = self.run_dir / "original_diffused.png"
        self.history: list[dict[str, Any]] = []
        self.best_record: dict[str, Any] | None = None
        self.evaluation_count = 0
        self.initial_vector = _initial_vector(config, self.rng, randomize=False)
        self.original_array: np.ndarray | None = None
        self.original_diffused_array: np.ndarray | None = None
        self.identity_models = _objective_models(config)
        self.strict_identity = _strict_identity(config)
        self.original_identity_cache: dict[str, dict[str, Any]] = {}
        self.original_diffused_identity_cache: dict[str, dict[str, Any]] = {}
        self.original_vae: np.ndarray | None = None
        self.original_diffused_vae: np.ndarray | None = None
        self.original_clip: np.ndarray | None = None
        self.original_diffused_clip: np.ndarray | None = None
        self.embedding_backends: dict[str, Any] = {}

    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.run_dir / "embedding_loss_config.json", self.config.raw)

        original_pil = load_pil_image(self.config.input_path)
        save_pil(self.original_path, original_pil)
        if self.config.diffusion.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        original_diffused = edit_image(original_pil, self.config.prompt, self.config.diffusion)
        save_pil(self.original_diffused_path, original_diffused)

        self.original_array = load_image(self.original_path)
        self.original_diffused_array = load_image(self.original_diffused_path)
        self._cache_original_backends()

    def _cache_original_backends(self) -> None:
        identity_config = _metric_config(self.config.objective, "identity")
        if _enabled(identity_config, True):
            if bool(identity_config.get("use_pre_identity", True)):
                self.original_identity_cache = _cache_identity_embeddings(self.original_path, self.config.identity, self.identity_models, self.strict_identity)
            if bool(identity_config.get("use_post_identity", True)):
                self.original_diffused_identity_cache = _cache_identity_embeddings(self.original_diffused_path, self.config.identity, self.identity_models, self.strict_identity)
            self.embedding_backends["identity"] = {
                "models": self.identity_models,
                "detector_backend": self.config.identity.detector_backend,
                "direct_embedding": True,
                "fallback": "verify_distance",
            }

        vae_config = _metric_config(self.config.objective, "vae_latent")
        if _enabled(vae_config, True):
            if bool(vae_config.get("use_input_vae", False)):
                self.original_vae, info = _vae_latent(self.original_path, self.config.diffusion)
                self.embedding_backends["input_vae"] = info
            if bool(vae_config.get("use_output_vae", True)):
                self.original_diffused_vae, info = _vae_latent(self.original_diffused_path, self.config.diffusion)
                self.embedding_backends["output_vae"] = info

        clip_config = _metric_config(self.config.objective, "clip_image")
        if _enabled(clip_config, False):
            model_id = str(clip_config.get("model_id", "openai/clip-vit-base-patch32"))
            if bool(clip_config.get("use_input_clip", False)):
                self.original_clip, info = _clip_embedding(self.original_path, model_id, self.config.diffusion)
                self.embedding_backends["input_clip"] = info
            if bool(clip_config.get("use_output_clip", True)):
                self.original_diffused_clip, info = _clip_embedding(self.original_diffused_path, model_id, self.config.diffusion)
                self.embedding_backends["output_clip"] = info

    def evaluate(self, vector: np.ndarray, label: str) -> dict[str, Any]:
        self.evaluation_count += 1
        iteration_name = f"iter_{self.evaluation_count:06d}"
        iteration_dir = self.iterations_dir / iteration_name
        iteration_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()

        clipped = np.clip(vector.astype(np.float64), 0.0, 1.0)
        parameters = _parameters_from_vector(self.config, clipped)
        steps = _steps_from_parameters(self.config, parameters, self.config.seed + self.evaluation_count)
        assert self.original_array is not None
        assert self.original_diffused_array is not None

        perturbed_array = apply_perturbation_pipeline(self.original_array, steps)
        perturbed_path = iteration_dir / "perturbed.png"
        perturbed_diffused_path = iteration_dir / "perturbed_diffused.png"
        save_image(perturbed_path, perturbed_array)
        perturbed_diffused = edit_image(load_pil_image(perturbed_path), self.config.prompt, self.config.diffusion)
        save_pil(perturbed_diffused_path, perturbed_diffused)
        perturbed_diffused_array = load_image(perturbed_diffused_path)

        errors: dict[str, Any] = {}
        metrics = self._metrics_for_candidate(perturbed_array, perturbed_path, perturbed_diffused_array, perturbed_diffused_path, errors)
        loss, components = _loss_from_embedding_metrics(
            self.config.objective,
            metrics,
            clipped,
            self.initial_vector,
        )
        selected_model = selected_diffusion_model(self.config.diffusion)
        record = {
            "iteration": self.evaluation_count,
            "label": label,
            "loss": loss,
            "loss_components": components,
            "metrics": metrics,
            "parameters": parameters,
            "normalized_parameters": clipped.tolist(),
            "perturbations": [perturbation_to_report(step) for step in steps],
            "diffusion": {
                **asdict(self.config.diffusion),
                "used_model": selected_model["name"],
                "used_model_id": selected_model["model_id"],
                "resolved_device": resolve_device(self.config.diffusion),
            },
            "embedding_backends": self.embedding_backends,
            "errors": errors,
            "outputs": {
                "perturbed": str(perturbed_path),
                "perturbed_diffused": str(perturbed_diffused_path),
                "metrics": str(iteration_dir / "metrics.json"),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(iteration_dir / "metrics.json", record)
        self.history.append(
            {
                "iteration": self.evaluation_count,
                "label": label,
                "loss": loss,
                "metrics": metrics,
                "parameters": parameters,
                "path": str(iteration_dir),
            }
        )
        if self.best_record is None or loss < float(self.best_record["loss"]):
            self._write_best(record, perturbed_path, perturbed_diffused_path)
        save_every = int(self.config.optimizer.get("save_every", 1) or 0)
        if save_every > 0 and self.evaluation_count % save_every == 0:
            self.write_history()
        return record

    def _metrics_for_candidate(
        self,
        perturbed_array: np.ndarray,
        perturbed_path: Path,
        perturbed_diffused_array: np.ndarray,
        perturbed_diffused_path: Path,
        errors: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.original_array is not None
        assert self.original_diffused_array is not None
        metrics: dict[str, Any] = {
            "input_stealth": _input_stealth_metrics(
                self.original_array,
                perturbed_array,
                self.original_path,
                perturbed_path,
                self.config.objective,
            ),
            "output_disruption": _output_disruption_metrics(
                self.original_diffused_array,
                perturbed_diffused_array,
                self.original_diffused_path,
                perturbed_diffused_path,
                self.config.objective,
            ),
        }

        identity_config = _metric_config(self.config.objective, "identity")
        if _enabled(identity_config, True):
            identity_metrics: dict[str, Any] = {}
            if bool(identity_config.get("use_pre_identity", True)):
                identity_metrics["pre"] = _identity_pair_metrics(
                    self.original_path,
                    perturbed_path,
                    self.config.identity,
                    self.identity_models,
                    self.original_identity_cache,
                    self.strict_identity,
                )
            if bool(identity_config.get("use_post_identity", True)):
                identity_metrics["post"] = _identity_pair_metrics(
                    self.original_diffused_path,
                    perturbed_diffused_path,
                    self.config.identity,
                    self.identity_models,
                    self.original_diffused_identity_cache,
                    self.strict_identity,
                )
            metrics["identity"] = identity_metrics

        vae_config = _metric_config(self.config.objective, "vae_latent")
        if _enabled(vae_config, True):
            metrics["vae_latent"] = {}
            if bool(vae_config.get("use_input_vae", False)):
                candidate, info = _vae_latent(perturbed_path, self.config.diffusion)
                distance, status = _pair_distance(self.original_vae, candidate, bool(vae_config.get("strict", False)), "input VAE")
                metrics["vae_latent"]["input"] = distance
                if not status.get("ok"):
                    errors["input_vae"] = status | {"candidate": info}
            if bool(vae_config.get("use_output_vae", True)):
                candidate, info = _vae_latent(perturbed_diffused_path, self.config.diffusion)
                distance, status = _pair_distance(self.original_diffused_vae, candidate, bool(vae_config.get("strict", False)), "output VAE")
                metrics["vae_latent"]["output"] = distance
                if not status.get("ok"):
                    errors["output_vae"] = status | {"candidate": info}

        clip_config = _metric_config(self.config.objective, "clip_image")
        if _enabled(clip_config, False):
            model_id = str(clip_config.get("model_id", "openai/clip-vit-base-patch32"))
            metrics["clip_image"] = {}
            if bool(clip_config.get("use_input_clip", False)):
                candidate, info = _clip_embedding(perturbed_path, model_id, self.config.diffusion)
                distance, status = _pair_distance(self.original_clip, candidate, bool(clip_config.get("strict", False)), "input CLIP")
                metrics["clip_image"]["input"] = distance
                if not status.get("ok"):
                    errors["input_clip"] = status | {"candidate": info}
            if bool(clip_config.get("use_output_clip", True)):
                candidate, info = _clip_embedding(perturbed_diffused_path, model_id, self.config.diffusion)
                distance, status = _pair_distance(self.original_diffused_clip, candidate, bool(clip_config.get("strict", False)), "output CLIP")
                metrics["clip_image"]["output"] = distance
                if not status.get("ok"):
                    errors["output_clip"] = status | {"candidate": info}

        return metrics

    def _write_best(self, record: dict[str, Any], perturbed_path: Path, perturbed_diffused_path: Path) -> None:
        self.best_record = deepcopy(record)
        shutil.copy2(perturbed_path, self.best_dir / "perturbed.png")
        shutil.copy2(perturbed_diffused_path, self.best_dir / "perturbed_diffused.png")
        _write_json(self.best_dir / "report.json", record)

    def write_history(self) -> None:
        _write_json(self.run_dir / "embedding_loss_history.json", {"evaluations": self.history})

    def final_report(self, status: str) -> dict[str, Any]:
        best = self.best_record or {}
        selected_model = selected_diffusion_model(self.config.diffusion)
        report = {
            "status": status,
            "config_path": str(self.config.config_path),
            "input": str(self.config.input_path),
            "prompt": self.config.prompt,
            "output_dir": str(self.run_dir),
            "configured_seed": self.config.configured_seed,
            "seed": self.config.seed,
            "randomize_seed": self.config.randomize_seed,
            "random_seed_range": list(self.config.random_seed_range),
            "best_iteration": best.get("iteration"),
            "best_parameters": best.get("parameters"),
            "best_loss": best.get("loss"),
            "best_metrics": best.get("metrics"),
            "optimizer": self.config.optimizer,
            "objective": self.config.objective,
            "diffusion": best.get("diffusion") or {
                **asdict(self.config.diffusion),
                "used_model": selected_model["name"],
                "used_model_id": selected_model["model_id"],
                "resolved_device": resolve_device(self.config.diffusion),
            },
            "embedding_backends": self.embedding_backends,
            "evaluations": len(self.history),
            "outputs": {
                "embedding_loss_config": str(self.run_dir / "embedding_loss_config.json"),
                "original": str(self.original_path),
                "original_diffused": str(self.original_diffused_path),
                "best": str(self.best_dir),
                "iterations": str(self.iterations_dir),
                "embedding_loss_history": str(self.run_dir / "embedding_loss_history.json"),
                "report": str(self.run_dir / "report.json"),
            },
            "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in self.history),
        }
        _write_json(self.run_dir / "report.json", report)
        return report


def _output_disruption_score(metrics: dict[str, Any]) -> float | None:
    disruption = metrics.get("output_disruption")
    if not isinstance(disruption, dict):
        return None
    values = [
        float(disruption[key])
        for key in ("pixel_l2", "ssim_drop", "lpips")
        if disruption.get(key) is not None
    ]
    return sum(values) if values else None


def _stop_reached(config: EmbeddingLossConfig, best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    stop = config.optimizer.get("stop", {})
    if not isinstance(stop, dict):
        stop = {}
    if stop.get("loss_below") is not None and float(best.get("loss", math.inf)) <= float(stop["loss_below"]):
        return True
    metrics = best.get("metrics") if isinstance(best, dict) else None
    if not isinstance(metrics, dict):
        return False
    post_distance_above = stop.get("post_identity_distance_above")
    if post_distance_above is not None:
        post = metrics.get("identity", {}).get("post", {}).get("mean_distance_percent")
        if post is not None and float(post) >= float(post_distance_above):
            return True
    output_disruption_above = stop.get("output_disruption_above")
    if output_disruption_above is not None:
        score = _output_disruption_score(metrics)
        if score is not None and score >= float(output_disruption_above):
            return True
    return False


def _run_random_search(runner: EmbeddingLossRunner) -> None:
    config = runner.config
    iterations = int(config.optimizer.get("iterations", 20))
    random_restarts = max(1, int(config.optimizer.get("random_restarts", 1)))
    patience = int(config.optimizer.get("patience", 0))
    best_loss = math.inf
    no_improve = 0
    for restart in range(random_restarts):
        initial = _initial_vector(config, runner.rng, randomize=(restart > 0 or config.initialization == "random"))
        runner.evaluate(initial, f"restart_{restart}_initial")
        for iteration in range(iterations):
            vector = np.asarray([runner.rng.random() for _ in config.parameter_specs], dtype=np.float64)
            record = runner.evaluate(vector, f"restart_{restart}_random_{iteration + 1}")
            if float(record["loss"]) + 1e-12 < best_loss:
                best_loss = float(record["loss"])
                no_improve = 0
            else:
                no_improve += 1
            if _stop_reached(config, runner.best_record):
                return
            if patience > 0 and no_improve >= patience:
                return


def _run_spsa(runner: EmbeddingLossRunner) -> None:
    config = runner.config
    iterations = int(config.optimizer.get("iterations", 20))
    random_restarts = max(1, int(config.optimizer.get("random_restarts", 1)))
    learning_rate = float(config.optimizer.get("learning_rate", 0.08))
    delta_scale = float(config.optimizer.get("spsa_delta", config.optimizer.get("finite_difference_epsilon", 0.08)))
    patience = int(config.optimizer.get("patience", 0))
    best_loss = math.inf
    no_improve = 0

    for restart in range(random_restarts):
        vector = _initial_vector(config, runner.rng, randomize=(restart > 0 or config.initialization == "random"))
        current = runner.evaluate(vector, f"restart_{restart}_initial")
        if float(current["loss"]) < best_loss:
            best_loss = float(current["loss"])
        for iteration in range(1, iterations + 1):
            ak = learning_rate / (iteration ** 0.602)
            ck = delta_scale / (iteration ** 0.101)
            direction = runner.np_rng.choice([-1.0, 1.0], size=len(config.parameter_specs))
            plus = np.clip(vector + ck * direction, 0.0, 1.0)
            minus = np.clip(vector - ck * direction, 0.0, 1.0)
            plus_record = runner.evaluate(plus, f"restart_{restart}_spsa_{iteration}_plus")
            minus_record = runner.evaluate(minus, f"restart_{restart}_spsa_{iteration}_minus")
            gradient = (float(plus_record["loss"]) - float(minus_record["loss"])) / max(2.0 * ck, 1e-12) * direction
            vector = np.clip(vector - ak * gradient, 0.0, 1.0)
            current = runner.evaluate(vector, f"restart_{restart}_spsa_{iteration}_current")
            if float(current["loss"]) + 1e-12 < best_loss:
                best_loss = float(current["loss"])
                no_improve = 0
            else:
                no_improve += 1
            if _stop_reached(config, runner.best_record):
                return
            if patience > 0 and no_improve >= patience:
                return


def run_embedding_loss_pipeline(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_embedding_loss_config(config_path)
    runner = EmbeddingLossRunner(config)
    status = "completed"
    report: dict[str, Any] | None = None
    try:
        runner.prepare()
        optimizer_type = str(config.optimizer.get("type", "spsa")).strip().lower()
        if optimizer_type == "random_search":
            _run_random_search(runner)
        elif optimizer_type == "spsa":
            _run_spsa(runner)
        else:
            raise ValueError("embedding loss optimizer.type must be 'spsa' or 'random_search'")
    except Exception:
        status = "failed"
        raise
    finally:
        runner.write_history()
        if runner.best_record is not None:
            report = runner.final_report(status)
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_json(runner.run_dir / "report.json", report)
    if report is None:
        raise RuntimeError("embedding loss run finished without producing any evaluations")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run geometric-v1 embedding-loss optimization")
    parser.add_argument("--config", type=Path, default=Path("embedding_loss.json"))
    args = parser.parse_args(argv)
    report = run_embedding_loss_pipeline(args.config)
    print(json.dumps(report["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
