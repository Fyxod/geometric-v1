from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .diffusion import edit_image, resolve_device, selected_diffusion_model
from .events import EventCallback, emit_event
from .image_io import load_image, load_pil_image, save_image, save_pil
from .perturbations import apply_perturbation_pipeline


def run_perturb_only(config_path: Path, event_callback: EventCallback | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    emit_event(event_callback, "run_started", run_type="perturb", config_path=str(config_path))
    try:
        config = load_pipeline_config(config_path)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        original_path = config.output_dir / "original.png"
        perturbed_path = config.output_dir / "perturbed.png"
        report_path = config.output_dir / "report.json"

        original_pil = load_pil_image(config.input_path)
        save_pil(original_path, original_pil)
        emit_event(event_callback, "image_written", name="original", path=str(original_path))

        emit_event(event_callback, "perturbation_started", input_path=str(config.input_path), output_path=str(perturbed_path))
        original_array = load_image(config.input_path)
        perturbed_array = apply_perturbation_pipeline(original_array, config.perturbations)
        save_image(perturbed_path, perturbed_array)
        emit_event(event_callback, "perturbation_completed", output_path=str(perturbed_path))
        emit_event(event_callback, "image_written", name="perturbed", path=str(perturbed_path))

        report = {
            "config_path": str(config.config_path),
            "input": str(config.input_path),
            "output_dir": str(config.output_dir),
            "seed": config.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "perturbations": [asdict(step) for step in config.perturbations],
            "outputs": {
                "original": str(original_path),
                "perturbed": str(perturbed_path),
                "report": str(report_path),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        emit_event(event_callback, "image_written", name="report", path=str(report_path))
        emit_event(event_callback, "run_completed", run_type="perturb", report_path=str(report_path))
        return report
    except Exception as exc:
        emit_event(event_callback, "run_failed", run_type="perturb", error=f"{type(exc).__name__}: {exc}")
        raise


def run_diffuse_only(config_path: Path, event_callback: EventCallback | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    emit_event(event_callback, "run_started", run_type="diffuse", config_path=str(config_path))
    try:
        config = load_pipeline_config(config_path)
        if config.diffusion.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        resolved_device = resolve_device(config.diffusion)
        selected_model = selected_diffusion_model(config.diffusion)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        original_path = config.output_dir / "original.png"
        diffused_path = config.output_dir / "diffused.png"
        report_path = config.output_dir / "report.json"

        original_pil = load_pil_image(config.input_path)
        save_pil(original_path, original_pil)
        emit_event(event_callback, "image_written", name="original", path=str(original_path))

        emit_event(
            event_callback,
            "diffusion_started",
            mode="single",
            prompt=config.prompt,
            device=resolved_device,
            selected_model=selected_model["name"],
            model_id=selected_model["model_id"],
            output_path=str(diffused_path),
        )
        diffused = edit_image(original_pil, config.prompt, config.diffusion)
        save_pil(diffused_path, diffused)
        emit_event(event_callback, "diffusion_completed", output_paths=[str(diffused_path)])
        emit_event(event_callback, "image_written", name="diffused", path=str(diffused_path))

        report = {
            "config_path": str(config.config_path),
            "input": str(config.input_path),
            "output_dir": str(config.output_dir),
            "prompt": config.prompt,
            "seed": config.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "diffusion": {
                **asdict(config.diffusion),
                "used_model": selected_model["name"],
                "used_model_id": selected_model["model_id"],
                "resolved_device": resolved_device,
                "execution_mode": "single",
            },
            "outputs": {
                "original": str(original_path),
                "diffused": str(diffused_path),
                "report": str(report_path),
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        emit_event(event_callback, "image_written", name="report", path=str(report_path))
        emit_event(event_callback, "run_completed", run_type="diffuse", report_path=str(report_path))
        return report
    except Exception as exc:
        emit_event(event_callback, "run_failed", run_type="diffuse", error=f"{type(exc).__name__}: {exc}")
        raise
