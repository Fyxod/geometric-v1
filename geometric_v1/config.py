from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DEEPFACE_MODELS = {
    "VGG-Face": True,
    "Facenet": True,
    "Facenet512": True,
    "OpenFace": True,
    "DeepFace": True,
    "DeepID": True,
    "ArcFace": True,
    "Dlib": False,
    "SFace": True,
    "GhostFaceNet": True,
    "Buffalo_L": False,
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
class DiffusionConfig:
    model_id: str = "timbrooks/instruct-pix2pix"
    device: str = "auto"
    gpu_index: int = 0
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.0
    max_size: int = 512
    seed: int = 7


@dataclass(frozen=True)
class DeepFaceConfig:
    enabled: bool = True
    detector_backend: str = "skip"
    distance_metric: str = "cosine"
    enforce_detection: bool = False
    align: bool = False
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


def _diffusion_from_dict(values: dict[str, Any], base_seed: int) -> DiffusionConfig:
    return DiffusionConfig(
        model_id=str(values.get("model_id", "timbrooks/instruct-pix2pix")),
        device=str(values.get("device", "auto")),
        gpu_index=int(values.get("gpu_index", 0)),
        num_inference_steps=int(values.get("num_inference_steps", 10)),
        guidance_scale=float(values.get("guidance_scale", 7.5)),
        image_guidance_scale=float(values.get("image_guidance_scale", 1.0)),
        max_size=int(values.get("max_size", 512)),
        seed=int(values.get("seed", base_seed)),
    )


def _deepface_from_dict(values: dict[str, Any]) -> DeepFaceConfig:
    models = dict(DEFAULT_DEEPFACE_MODELS)
    models.update({str(key): bool(value) for key, value in values.get("models", {}).items()})
    return DeepFaceConfig(
        enabled=bool(values.get("enabled", True)),
        detector_backend=str(values.get("detector_backend", "skip")),
        distance_metric=str(values.get("distance_metric", "cosine")),
        enforce_detection=bool(values.get("enforce_detection", False)),
        align=bool(values.get("align", False)),
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
