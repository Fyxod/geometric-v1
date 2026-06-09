from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DEEPFACE_MODELS = {
    "SFace": True,
    "OpenFace": True,
    "Facenet": True,
    "Facenet512": True,
}


@dataclass(frozen=True)
class PerturbationStep:
    method: str
    enabled: bool = True
    strength: float = 0.0
    seed: int = 7
    grid: int = 7
    coefficients: int = 12
    sigma: float = 12.0
    rolling_frequency: float = 1.5
    rolling_phase: float = 0.0
    rolling_shear: float = 0.03
    rolling_acceleration: float = 0.0


@dataclass(frozen=True)
class InstructPix2PixConfig:
    enabled: bool = True
    model_id: str = "timbrooks/instruct-pix2pix"
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.0
    max_size: int = 512
    seed: int = 7


@dataclass(frozen=True)
class Flux2KleinConfig:
    enabled: bool = False
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    max_size: int = 768
    height: int | None = None
    width: int | None = None
    max_sequence_length: int = 512
    text_encoder_out_layers: tuple[int, int, int] = (9, 18, 27)
    torch_dtype: str = "bfloat16"
    cpu_offload: bool = False
    sigmas: list[float] | None = None
    seed: int = 7


@dataclass(frozen=True)
class DiffusionConfig:
    cpu: bool = False
    device: str = "auto"
    gpu_index: int = 0
    selected_model: str = "instruct_pix2pix"
    selected_model_id: str = "timbrooks/instruct-pix2pix"
    model_id: str = "timbrooks/instruct-pix2pix"
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.0
    max_size: int = 512
    seed: int = 7
    instruct_pix2pix: InstructPix2PixConfig = field(default_factory=InstructPix2PixConfig)
    flux2_klein: Flux2KleinConfig = field(default_factory=Flux2KleinConfig)


@dataclass(frozen=True)
class DeepFaceConfig:
    enabled: bool = True
    detector_backend: str = "skip"
    distance_metric: str = "cosine"
    enforce_detection: bool = False
    align: bool = False
    workers: int | str = "auto"
    models: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_DEEPFACE_MODELS))


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    input_path: Path
    output_dir: Path
    prompt: str
    seed: int
    perturbations: list[PerturbationStep]
    diffusion: DiffusionConfig
    deepface: DeepFaceConfig
    raw: dict[str, Any]


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _perturbation_from_dict(values: dict[str, Any], base_seed: int, index: int) -> PerturbationStep:
    strength = float(values.get("strength", 0.0))
    return PerturbationStep(
        method=str(values["method"]),
        enabled=bool(values.get("enabled", True)),
        strength=strength,
        seed=int(values.get("seed", base_seed + index)),
        grid=int(values.get("grid", 7)),
        coefficients=int(values.get("coefficients", 12)),
        sigma=float(values.get("sigma", 12.0)),
        rolling_frequency=float(values.get("rolling_frequency", values.get("frequency", 1.5))),
        rolling_phase=float(values.get("rolling_phase", values.get("phase", 0.0))),
        rolling_shear=float(values.get("rolling_shear", values.get("shear", 0.03))),
        rolling_acceleration=float(values.get("rolling_acceleration", values.get("acceleration", 0.0))),
    )


def _instruct_pix2pix_from_dict(values: dict[str, Any], base_seed: int) -> InstructPix2PixConfig:
    return InstructPix2PixConfig(
        enabled=bool(values.get("enabled", True)),
        model_id=str(values.get("model_id", "timbrooks/instruct-pix2pix")),
        num_inference_steps=int(values.get("num_inference_steps", 10)),
        guidance_scale=float(values.get("guidance_scale", 7.5)),
        image_guidance_scale=float(values.get("image_guidance_scale", 1.0)),
        max_size=int(values.get("max_size", 512)),
        seed=int(values.get("seed", base_seed)),
    )


def _flux2_klein_from_dict(values: dict[str, Any], base_seed: int) -> Flux2KleinConfig:
    layers = values.get("text_encoder_out_layers", [9, 18, 27])
    if not isinstance(layers, (list, tuple)) or len(layers) != 3:
        raise ValueError("flux2_klein.text_encoder_out_layers must be a three-item list")
    sigmas = values.get("sigmas")
    return Flux2KleinConfig(
        enabled=bool(values.get("enabled", False)),
        model_id=str(values.get("model_id", "black-forest-labs/FLUX.2-klein-4B")),
        num_inference_steps=int(values.get("num_inference_steps", 4)),
        guidance_scale=float(values.get("guidance_scale", 1.0)),
        max_size=int(values.get("max_size", 768)),
        height=int(values["height"]) if values.get("height") is not None else None,
        width=int(values["width"]) if values.get("width") is not None else None,
        max_sequence_length=int(values.get("max_sequence_length", 512)),
        text_encoder_out_layers=(int(layers[0]), int(layers[1]), int(layers[2])),
        torch_dtype=str(values.get("torch_dtype", "bfloat16")),
        cpu_offload=bool(values.get("cpu_offload", False)),
        sigmas=[float(value) for value in sigmas] if isinstance(sigmas, list) else None,
        seed=int(values.get("seed", base_seed)),
    )


def _diffusion_from_dict(values: dict[str, Any], base_seed: int) -> DiffusionConfig:
    models = values.get("models")
    diffusion_seed = int(values.get("seed", base_seed))
    if isinstance(models, dict):
        instruct_values = models.get("instruct_pix2pix")
        flux_values = models.get("flux2_klein")
        instruct = _instruct_pix2pix_from_dict(
            instruct_values if isinstance(instruct_values, dict) else {"enabled": False},
            diffusion_seed,
        )
        flux = _flux2_klein_from_dict(
            flux_values if isinstance(flux_values, dict) else {"enabled": False},
            diffusion_seed,
        )
    else:
        instruct = _instruct_pix2pix_from_dict(values, diffusion_seed)
        flux = _flux2_klein_from_dict({}, diffusion_seed)

    if flux.enabled:
        selected_model = "flux2_klein"
        selected_model_id = flux.model_id
        active_seed = flux.seed
        active_steps = flux.num_inference_steps
        active_guidance = flux.guidance_scale
        active_image_guidance = 1.0
        active_max_size = flux.max_size
    elif instruct.enabled:
        selected_model = "instruct_pix2pix"
        selected_model_id = instruct.model_id
        active_seed = instruct.seed
        active_steps = instruct.num_inference_steps
        active_guidance = instruct.guidance_scale
        active_image_guidance = instruct.image_guidance_scale
        active_max_size = instruct.max_size
    else:
        raise ValueError("At least one diffusion model block must be enabled")

    return DiffusionConfig(
        cpu=bool(values.get("cpu", False)),
        device=str(values.get("device", "auto")),
        gpu_index=int(values.get("gpu_index", 0)),
        selected_model=selected_model,
        selected_model_id=selected_model_id,
        model_id=selected_model_id,
        num_inference_steps=active_steps,
        guidance_scale=active_guidance,
        image_guidance_scale=active_image_guidance,
        max_size=active_max_size,
        seed=active_seed,
        instruct_pix2pix=instruct,
        flux2_klein=flux,
    )


def _deepface_from_dict(values: dict[str, Any]) -> DeepFaceConfig:
    models = dict(DEFAULT_DEEPFACE_MODELS)
    configured_models = values.get("models", {})
    if isinstance(configured_models, dict):
        for key, value in configured_models.items():
            model_name = str(key)
            if model_name in models:
                models[model_name] = bool(value)
    workers_value = values.get("workers", "auto")
    workers: int | str
    if isinstance(workers_value, int):
        workers = workers_value
    else:
        workers_text = str(workers_value)
        workers = int(workers_text) if workers_text.isdigit() else workers_text
    return DeepFaceConfig(
        enabled=bool(values.get("enabled", True)),
        detector_backend=str(values.get("detector_backend", "skip")),
        distance_metric=str(values.get("distance_metric", "cosine")),
        enforce_detection=bool(values.get("enforce_detection", False)),
        align=bool(values.get("align", False)),
        workers=workers,
        models=models,
    )


def load_pipeline_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    seed = int(data.get("seed", 7))
    perturbations = [
        _perturbation_from_dict(item, seed, index)
        for index, item in enumerate(data.get("perturbations", []))
    ]
    return PipelineConfig(
        config_path=path,
        input_path=_resolve(base_dir, str(data["input"])),
        output_dir=_resolve(base_dir, str(data.get("output_dir", "output/run_001"))),
        prompt=str(data["prompt"]),
        seed=seed,
        perturbations=perturbations,
        diffusion=_diffusion_from_dict(data.get("diffusion", {}), seed),
        deepface=_deepface_from_dict(data.get("deepface", {})),
        raw=data,
    )
