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
from scipy.linalg import sqrtm
from scipy.ndimage import gaussian_filter

from .config import (
    DEFAULT_DEEPFACE_MODELS,
    DeepFaceConfig,
    DiffusionConfig,
    PerturbationStep,
    _deepface_from_dict,
    _diffusion_from_dict,
    _perturbation_from_dict,
)
from .deepface_compare import compare_images
from .diffusion import edit_image, resolve_device, selected_diffusion_model
from .image_io import load_image, load_pil_image, save_image, save_pil
from .perturbations import apply_perturbation_pipeline


INTEGER_FIELDS = {"grid", "coefficients"}
PARAMETER_FIELDS = {
    "strength",
    "grid",
    "coefficients",
    "sigma",
    "rolling_frequency",
    "rolling_phase",
    "rolling_shear",
    "rolling_acceleration",
}

_LPIPS_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class ParameterSpec:
    method: str
    field: str
    lower: float
    upper: float

    @property
    def name(self) -> str:
        return f"{self.method}.{self.field}"

    @property
    def is_integer(self) -> bool:
        return self.field in INTEGER_FIELDS


@dataclass(frozen=True)
class LossConfig:
    config_path: Path
    raw: dict[str, Any]
    pipeline_config: Path | None
    input_path: Path
    prompt: str
    output_parent: Path
    seed: int
    perturbation_templates: list[PerturbationStep]
    diffusion: DiffusionConfig
    deepface: DeepFaceConfig
    optimizer: dict[str, Any]
    objective: dict[str, Any]
    initialization: str
    initial_values: dict[str, float]
    parameter_specs: list[ParameterSpec]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _nested_lookup(values: dict[str, Any], method: str, field: str) -> Any:
    flat_key = f"{method}.{field}"
    if flat_key in values:
        return values[flat_key]
    method_values = values.get(method)
    if isinstance(method_values, dict) and field in method_values:
        return method_values[field]
    return None


def _parse_bounds(values: dict[str, Any]) -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    for key, value in values.items():
        if isinstance(value, dict):
            method = str(key)
            for field, bounds in value.items():
                specs.append(_parameter_spec(method, str(field), bounds))
        else:
            if "." not in str(key):
                raise ValueError(f"parameter bound '{key}' must be nested by method or use method.field")
            method, field = str(key).rsplit(".", 1)
            specs.append(_parameter_spec(method, field, value))
    return specs


def _parameter_spec(method: str, field: str, bounds: Any) -> ParameterSpec:
    if field not in PARAMETER_FIELDS:
        raise ValueError(f"unsupported optimizable parameter field: {method}.{field}")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"bounds for {method}.{field} must be a two-item list")
    lower = float(bounds[0])
    upper = float(bounds[1])
    if lower > upper:
        raise ValueError(f"bounds for {method}.{field} have lower > upper")
    return ParameterSpec(method=method, field=field, lower=lower, upper=upper)


def _load_reference_pipeline(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _read_json(path)


def _config_block(
    data: dict[str, Any],
    reference: dict[str, Any],
    block_name: str,
) -> dict[str, Any]:
    block = data.get(block_name)
    reference_block = reference.get(block_name, {})
    if block is None:
        return deepcopy(reference_block) if isinstance(reference_block, dict) else {}
    if not isinstance(block, dict):
        raise ValueError(f"{block_name} must be an object")
    if block.get("from_pipeline", False):
        overrides = block.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{block_name}.overrides must be an object")
        return _deep_update(reference_block if isinstance(reference_block, dict) else {}, overrides)
    return block


def load_loss_config(path: Path) -> LossConfig:
    path = path.resolve()
    data = _read_json(path)
    base_dir = path.parent
    pipeline_config = data.get("pipeline_config")
    pipeline_path = _resolve(base_dir, str(pipeline_config)).resolve() if pipeline_config else None
    reference = _load_reference_pipeline(pipeline_path)

    seed = int(data.get("seed", reference.get("seed", 7)))
    input_value = data.get("input", reference.get("input"))
    if input_value is None:
        raise ValueError("loss config must include input or pipeline_config with input")
    prompt = str(data.get("prompt", reference.get("prompt", "")))
    if not prompt:
        raise ValueError("loss config must include prompt or pipeline_config with prompt")

    perturbation_items = data.get("perturbations", reference.get("perturbations", []))
    if not isinstance(perturbation_items, list) or not perturbation_items:
        raise ValueError("loss config must include at least one perturbation template")
    perturbations = [
        _perturbation_from_dict(item, seed, index)
        for index, item in enumerate(perturbation_items)
    ]

    diffusion_values = _config_block(data, reference, "diffusion")
    deepface_values = _config_block(data, reference, "deepface")

    parameter_config = data.get("parameters", {})
    if not isinstance(parameter_config, dict):
        raise ValueError("parameters must be an object")
    bounds = parameter_config.get("bounds", {})
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError("parameters.bounds must include at least one optimizable parameter")
    specs = _parse_bounds(bounds)
    initial_values = parameter_config.get("initial_values", {})
    if not isinstance(initial_values, dict):
        raise ValueError("parameters.initial_values must be an object")

    return LossConfig(
        config_path=path,
        raw=data,
        pipeline_config=pipeline_path,
        input_path=_resolve(base_dir, str(input_value)).resolve(),
        prompt=prompt,
        output_parent=_resolve(base_dir, str(data.get("output_dir", "output"))).resolve(),
        seed=seed,
        perturbation_templates=perturbations,
        diffusion=_diffusion_from_dict(diffusion_values, seed),
        deepface=_deepface_from_dict(deepface_values),
        optimizer=data.get("optimizer", {}) if isinstance(data.get("optimizer", {}), dict) else {},
        objective=data.get("objective", {}) if isinstance(data.get("objective", {}), dict) else {},
        initialization=str(parameter_config.get("initialization", "fixed")),
        initial_values={str(key): float(value) for key, value in initial_values.items() if not isinstance(value, dict)},
        parameter_specs=specs,
    )


def _step_map(steps: list[PerturbationStep]) -> dict[str, PerturbationStep]:
    return {step.method: step for step in steps}


def _value_to_normalized(spec: ParameterSpec, value: float) -> float:
    if spec.upper == spec.lower:
        return 0.0
    return max(0.0, min(1.0, (value - spec.lower) / (spec.upper - spec.lower)))


def _normalized_to_value(spec: ParameterSpec, value: float) -> float | int:
    clipped = max(0.0, min(1.0, float(value)))
    raw = spec.lower + clipped * (spec.upper - spec.lower)
    if spec.is_integer:
        return int(round(raw))
    return float(raw)


def _initial_vector(config: LossConfig, rng: random.Random, randomize: bool = False) -> np.ndarray:
    values: list[float] = []
    step_by_method = _step_map(config.perturbation_templates)
    for spec in config.parameter_specs:
        configured = _nested_lookup(config.raw.get("parameters", {}).get("initial_values", {}), spec.method, spec.field)
        if configured is None:
            configured = config.initial_values.get(spec.name)
        if randomize or config.initialization == "random":
            normalized = rng.random()
        elif configured is not None:
            normalized = _value_to_normalized(spec, float(configured))
        else:
            step = step_by_method.get(spec.method)
            normalized = _value_to_normalized(spec, float(getattr(step, spec.field, spec.lower)))
        values.append(normalized)
    return np.asarray(values, dtype=np.float64)


def _parameters_from_vector(config: LossConfig, vector: np.ndarray) -> dict[str, dict[str, float | int]]:
    parameters: dict[str, dict[str, float | int]] = {}
    for spec, normalized in zip(config.parameter_specs, vector):
        parameters.setdefault(spec.method, {})[spec.field] = _normalized_to_value(spec, float(normalized))
    return parameters


def _steps_from_parameters(config: LossConfig, parameters: dict[str, dict[str, float | int]], seed: int) -> list[PerturbationStep]:
    steps: list[PerturbationStep] = []
    for index, template in enumerate(config.perturbation_templates):
        item = asdict(template)
        item["seed"] = seed + index
        for field, value in parameters.get(template.method, {}).items():
            item[field] = value
        steps.append(_perturbation_from_dict(item, seed, index))
    return steps


def _psnr(original: np.ndarray, perturbed: np.ndarray) -> float:
    mse = float(np.mean((original - perturbed) ** 2))
    if mse <= 1e-12:
        return math.inf
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _ssim(original: np.ndarray, perturbed: np.ndarray) -> float:
    # Compact RGB SSIM implementation to avoid another dependency.
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for channel in range(original.shape[2]):
        x = original[:, :, channel].astype(np.float64)
        y = perturbed[:, :, channel].astype(np.float64)
        mux = gaussian_filter(x, sigma=1.5)
        muy = gaussian_filter(y, sigma=1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = gaussian_filter(x * x, sigma=1.5) - mux2
        sigy2 = gaussian_filter(y * y, sigma=1.5) - muy2
        sigxy = gaussian_filter(x * y, sigma=1.5) - muxy
        numerator = (2.0 * muxy + c1) * (2.0 * sigxy + c2)
        denominator = (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def _fid_rgb(original: np.ndarray, perturbed: np.ndarray) -> float:
    # Single-image FID is weak. This uses per-pixel RGB Gaussian features, not Inception features.
    x = original.reshape(-1, original.shape[2]).astype(np.float64)
    y = perturbed.reshape(-1, perturbed.shape[2]).astype(np.float64)
    mux = np.mean(x, axis=0)
    muy = np.mean(y, axis=0)
    covx = np.cov(x, rowvar=False)
    covy = np.cov(y, rowvar=False)
    product = sqrtm(covx @ covy)
    if np.iscomplexobj(product):
        product = product.real
    diff = mux - muy
    return float(diff @ diff + np.trace(covx + covy - 2.0 * product))


def _lpips_distance(original_path: Path, perturbed_path: Path) -> tuple[float | None, str | None]:
    try:
        import torch
        from PIL import Image
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_fn = _get_lpips_model(device)

        def tensor(path: Path):
            image = Image.open(path).convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            value = loss_fn(tensor(original_path), tensor(perturbed_path))
        return float(value.item()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _get_lpips_model(device: str):
    loss_fn = _LPIPS_MODEL_CACHE.get(device)
    if loss_fn is not None:
        return loss_fn

    import lpips

    loss_fn = lpips.LPIPS(net="alex").to(device)
    loss_fn.eval()
    _LPIPS_MODEL_CACHE[device] = loss_fn
    return loss_fn


def _beta_metrics(
    original: np.ndarray,
    perturbed: np.ndarray,
    original_path: Path,
    perturbed_path: Path,
    objective: dict[str, Any],
) -> dict[str, Any]:
    if not bool(objective.get("use_beta", True)):
        return {}
    beta = objective.get("beta", {})
    if not isinstance(beta, dict):
        beta = {}
    if not _enabled(beta, True):
        return {}
    metrics: dict[str, Any] = {}
    if _beta_metric_enabled(beta, "psnr", True):
        metrics["psnr"] = _psnr(original, perturbed)
    if _beta_metric_enabled(beta, "ssim", True):
        metrics["ssim"] = _ssim(original, perturbed)
    if _beta_metric_enabled(beta, "fid", False):
        metrics["fid"] = _fid_rgb(original, perturbed)
    if _beta_metric_enabled(beta, "lpips", False):
        value, error = _lpips_distance(original_path, perturbed_path)
        metrics["lpips"] = value
        if error:
            metrics["lpips_error"] = error
    return metrics


def _mean_match(report: dict[str, Any]) -> float | None:
    summary = report.get("summary")
    if isinstance(summary, dict) and summary.get("mean_match_percent") is not None:
        return float(summary["mean_match_percent"])
    values = []
    for value in report.get("models", {}).values():
        if value.get("ok") and value.get("match_percent") is not None:
            values.append(float(value["match_percent"]))
    return sum(values) / len(values) if values else None


def _alpha_pre_enabled(objective: dict[str, Any]) -> bool:
    return bool(objective.get("use_alpha_pre", True)) and _enabled(_metric_config(objective, "alpha_pre"), True)


def _alpha_post_enabled(objective: dict[str, Any]) -> bool:
    return _enabled(_metric_config(objective, "alpha_post"), True)


def _metric_config(objective: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = objective
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _enabled(config: dict[str, Any], default: bool) -> bool:
    return bool(config.get("enabled", default))


def _beta_metric_enabled(beta_config: dict[str, Any], metric_name: str, default: bool) -> bool:
    use_key = f"use_{metric_name.replace('-', '_')}"
    if use_key in beta_config:
        return bool(beta_config[use_key])
    metric_config = beta_config.get(metric_name, {})
    if isinstance(metric_config, dict):
        return _enabled(metric_config, default)
    return default


def _weight(config: dict[str, Any], default: float) -> float:
    return float(config.get("weight", default))


def _loss_from_metrics(
    objective: dict[str, Any],
    metrics: dict[str, Any],
    vector: np.ndarray,
    initial_vector: np.ndarray,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    total = 0.0

    alpha_post_config = _metric_config(objective, "alpha_post")
    if _enabled(alpha_post_config, True):
        alpha_post = metrics.get("alpha_post")
        if alpha_post is None:
            components["alpha_post_missing"] = float(objective.get("missing_metric_penalty", 10000.0))
        else:
            components["alpha_post_objective"] = _weight(alpha_post_config, 1.0) * float(alpha_post)

    alpha_pre_config = _metric_config(objective, "alpha_pre")
    use_alpha_pre = bool(objective.get("use_alpha_pre", True)) and _enabled(alpha_pre_config, True)
    if use_alpha_pre:
        alpha_pre = metrics.get("alpha_pre")
        target = float(alpha_pre_config.get("target", alpha_pre_config.get("minimum", 90.0)))
        if alpha_pre is None:
            components["alpha_pre_missing"] = float(objective.get("missing_metric_penalty", 10000.0))
        else:
            violation = max(0.0, target - float(alpha_pre))
            components["alpha_pre_penalty"] = _weight(alpha_pre_config, 5.0) * violation * violation

    beta_config = _metric_config(objective, "beta")
    if bool(objective.get("use_beta", True)) and _enabled(beta_config, True):
        for metric_name, default_target, direction in (
            ("psnr", 30.0, "min"),
            ("ssim", 0.90, "min"),
            ("fid", 5.0, "max"),
            ("lpips", 0.25, "max"),
        ):
            metric_config = _metric_config(objective, "beta", metric_name)
            if not _beta_metric_enabled(beta_config, metric_name, metric_name in {"psnr", "ssim"}):
                continue
            value = metrics.get(metric_name)
            target = float(metric_config.get("target", metric_config.get("threshold", default_target)))
            weight = _weight(metric_config, 1.0)
            if value is None:
                components[f"{metric_name}_missing"] = float(objective.get("missing_metric_penalty", 10000.0))
                continue
            if direction == "min":
                violation = max(0.0, target - float(value))
            else:
                violation = max(0.0, float(value) - target)
            components[f"{metric_name}_penalty"] = weight * violation * violation

    reg_config = _metric_config(objective, "parameter_regularization")
    if _enabled(reg_config, True):
        diff = vector - initial_vector
        components["parameter_regularization_penalty"] = _weight(reg_config, 0.01) * float(diff @ diff)

    for value in components.values():
        total += float(value)
    return total, components


class LossRunner:
    def __init__(self, config: LossConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.output_parent / f"loss_run_{timestamp}"
        self.iterations_dir = self.run_dir / "iterations"
        self.best_dir = self.run_dir / "best"
        self.original_path = self.run_dir / "original.png"
        self.original_diffused_path = self.run_dir / "original_diffused.png"
        self.history: list[dict[str, Any]] = []
        self.best_record: dict[str, Any] | None = None
        self.evaluation_count = 0
        self.initial_vector = _initial_vector(config, self.rng, randomize=False)

    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.iterations_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.run_dir / "loss_config.json", self.config.raw)

        original_pil = load_pil_image(self.config.input_path)
        save_pil(self.original_path, original_pil)
        if self.config.diffusion.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        original_diffused = edit_image(original_pil, self.config.prompt, self.config.diffusion)
        save_pil(self.original_diffused_path, original_diffused)

    def evaluate(self, vector: np.ndarray, label: str) -> dict[str, Any]:
        self.evaluation_count += 1
        iteration_name = f"iter_{self.evaluation_count:06d}"
        iteration_dir = self.iterations_dir / iteration_name
        iteration_dir.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()

        clipped = np.clip(vector.astype(np.float64), 0.0, 1.0)
        parameters = _parameters_from_vector(self.config, clipped)
        steps = _steps_from_parameters(self.config, parameters, self.config.seed + self.evaluation_count)

        original_array = load_image(self.config.input_path)
        perturbed_array = apply_perturbation_pipeline(original_array, steps)
        perturbed_path = iteration_dir / "perturbed.png"
        perturbed_diffused_path = iteration_dir / "perturbed_diffused.png"
        save_image(perturbed_path, perturbed_array)
        perturbed_pil = load_pil_image(perturbed_path)
        perturbed_diffused = edit_image(perturbed_pil, self.config.prompt, self.config.diffusion)
        save_pil(perturbed_diffused_path, perturbed_diffused)

        deepface_enabled = self.config.deepface.enabled
        alpha_pre_report = None
        alpha_post_report = None
        alpha_pre = None
        alpha_post = None
        allow_parallel = bool(self.config.raw.get("deepface_allow_parallel", False))
        if deepface_enabled and _alpha_pre_enabled(self.config.objective):
            alpha_pre_report = compare_images(
                self.original_path,
                perturbed_path,
                self.config.deepface,
                allow_parallel=allow_parallel,
                event_context={"stage": "loss_alpha_pre", "iteration": self.evaluation_count},
            )
            alpha_pre = _mean_match(alpha_pre_report)
        if deepface_enabled and _alpha_post_enabled(self.config.objective):
            alpha_post_report = compare_images(
                self.original_diffused_path,
                perturbed_diffused_path,
                self.config.deepface,
                allow_parallel=allow_parallel,
                event_context={"stage": "loss_alpha_post", "iteration": self.evaluation_count},
            )
            alpha_post = _mean_match(alpha_post_report)

        beta_metrics = _beta_metrics(
            original_array,
            perturbed_array,
            self.original_path,
            perturbed_path,
            self.config.objective,
        )
        metrics = {
            "alpha_pre": alpha_pre,
            "alpha_post": alpha_post,
            **beta_metrics,
        }
        loss, components = _loss_from_metrics(
            self.config.objective,
            metrics,
            clipped,
            self.initial_vector,
        )
        selected_model = selected_diffusion_model(self.config.diffusion)
        metric_summary = {
            "alpha_pre": alpha_pre,
            "alpha_post": alpha_post,
            "beta": beta_metrics,
        }
        record = {
            "iteration": self.evaluation_count,
            "label": label,
            "loss": loss,
            "loss_components": components,
            "metric_summary": metric_summary,
            "parameters": parameters,
            "normalized_parameters": clipped.tolist(),
            "metrics": metrics,
            "alpha_pre_report": alpha_pre_report,
            "alpha_post_report": alpha_post_report,
            "perturbations": [asdict(step) for step in steps],
            "diffusion": {
                **asdict(self.config.diffusion),
                "used_model": selected_model["name"],
                "used_model_id": selected_model["model_id"],
                "resolved_device": resolve_device(self.config.diffusion),
            },
            "deepface_models": {
                name: enabled
                for name, enabled in self.config.deepface.models.items()
                if name in DEFAULT_DEEPFACE_MODELS
            },
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
                "alpha_pre": alpha_pre,
                "alpha_post": alpha_post,
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

    def _write_best(self, record: dict[str, Any], perturbed_path: Path, perturbed_diffused_path: Path) -> None:
        self.best_record = deepcopy(record)
        shutil.copy2(perturbed_path, self.best_dir / "perturbed.png")
        shutil.copy2(perturbed_diffused_path, self.best_dir / "perturbed_diffused.png")
        _write_json(self.best_dir / "report.json", record)

    def write_history(self) -> None:
        _write_json(self.run_dir / "loss_history.json", {"evaluations": self.history})

    def final_report(self, status: str) -> dict[str, Any]:
        best = self.best_record or {}
        report = {
            "status": status,
            "config_path": str(self.config.config_path),
            "input": str(self.config.input_path),
            "prompt": self.config.prompt,
            "output_dir": str(self.run_dir),
            "seed": self.config.seed,
            "best_iteration": best.get("iteration"),
            "best_parameters": best.get("parameters"),
            "best_alpha_pre": (best.get("metrics") or {}).get("alpha_pre") if isinstance(best.get("metrics"), dict) else None,
            "best_alpha_post": (best.get("metrics") or {}).get("alpha_post") if isinstance(best.get("metrics"), dict) else None,
            "best_metrics": best.get("metric_summary") or best.get("metrics"),
            "best_beta_metrics": {
                key: value
                for key, value in (best.get("metrics") or {}).items()
                if key not in {"alpha_pre", "alpha_post"}
            }
            if isinstance(best.get("metrics"), dict)
            else {},
            "final_loss": best.get("loss"),
            "optimizer": self.config.optimizer,
            "objective": self.config.objective,
            "diffusion": best.get("diffusion") or {
                **asdict(self.config.diffusion),
                **selected_diffusion_model(self.config.diffusion),
            },
            "deepface_models": {
                name: enabled
                for name, enabled in self.config.deepface.models.items()
                if name in DEFAULT_DEEPFACE_MODELS
            },
            "evaluations": len(self.history),
            "outputs": {
                "loss_config": str(self.run_dir / "loss_config.json"),
                "original": str(self.original_path),
                "original_diffused": str(self.original_diffused_path),
                "best": str(self.best_dir),
                "iterations": str(self.iterations_dir),
                "loss_history": str(self.run_dir / "loss_history.json"),
                "report": str(self.run_dir / "report.json"),
            },
        }
        _write_json(self.run_dir / "report.json", report)
        return report


def _stop_reached(config: LossConfig, best: dict[str, Any] | None) -> bool:
    if not best:
        return False
    stop = config.optimizer.get("stop", {})
    if not isinstance(stop, dict):
        stop = {}
    loss_below = stop.get("loss_below")
    if loss_below is not None and float(best.get("loss", math.inf)) <= float(loss_below):
        return True
    alpha_post_below = stop.get("alpha_post_below")
    metrics = best.get("metrics") if isinstance(best, dict) else None
    if alpha_post_below is not None and isinstance(metrics, dict):
        alpha_post = metrics.get("alpha_post")
        if alpha_post is not None and float(alpha_post) <= float(alpha_post_below):
            return True
    return False


def _run_random_search(runner: LossRunner) -> None:
    config = runner.config
    iterations = int(config.optimizer.get("iterations", 20))
    random_restarts = max(1, int(config.optimizer.get("random_restarts", 1)))
    no_improve = 0
    patience = int(config.optimizer.get("patience", 0))
    best_loss = math.inf
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


def _run_spsa(runner: LossRunner) -> None:
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


def _run_differential_evolution(runner: LossRunner) -> None:
    try:
        from scipy.optimize import differential_evolution
    except Exception as exc:
        raise RuntimeError(f"scipy differential_evolution is unavailable: {type(exc).__name__}: {exc}") from exc

    config = runner.config
    iterations = int(config.optimizer.get("iterations", 10))
    popsize = int(config.optimizer.get("popsize", 5))
    workers = int(config.optimizer.get("workers", 1))

    def objective(vector: np.ndarray) -> float:
        record = runner.evaluate(np.asarray(vector, dtype=np.float64), "differential_evolution")
        return float(record["loss"])

    bounds = [(0.0, 1.0)] * len(config.parameter_specs)
    differential_evolution(
        objective,
        bounds,
        maxiter=iterations,
        popsize=popsize,
        seed=config.seed,
        polish=False,
        workers=workers,
        updating="immediate" if workers == 1 else "deferred",
    )


def run_loss_pipeline(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_loss_config(config_path)
    runner = LossRunner(config)
    status = "completed"
    try:
        runner.prepare()
        optimizer_type = str(config.optimizer.get("type", "spsa")).strip().lower()
        if optimizer_type == "spsa":
            _run_spsa(runner)
        elif optimizer_type in {"random", "random_search"}:
            _run_random_search(runner)
        elif optimizer_type in {"differential_evolution", "scipy_differential_evolution"}:
            _run_differential_evolution(runner)
        else:
            raise ValueError(f"unsupported optimizer type: {optimizer_type}")
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        runner.write_history()
        if runner.best_record is not None:
            report = runner.final_report(status)
            report["elapsed_seconds"] = time.perf_counter() - started
            _write_json(runner.run_dir / "report.json", report)
    if runner.best_record is None:
        raise RuntimeError("loss pipeline did not evaluate any candidates")
    report = runner.final_report(status)
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(runner.run_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run loss-guided black-box perturbation optimization")
    parser.add_argument("--config", type=Path, default=Path("loss.json"))
    args = parser.parse_args(argv)
    report = run_loss_pipeline(args.config)
    print(json.dumps(report["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
