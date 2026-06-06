from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import load_pipeline_config
from .deepface_compare import compare_images
from .diffusion import edit_image, edit_images, resolve_device
from .image_io import load_image, load_pil_image, save_image, save_pil
from .perturbations import apply_perturbation_pipeline


def run_pipeline(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_pipeline_config(config_path)
    if config.diffusion.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    resolved_device = resolve_device(config.diffusion)
    use_gpu_optimizations = resolved_device.startswith("cuda")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    original_path = config.output_dir / "original.png"
    perturbed_path = config.output_dir / "perturbed.png"
    original_diffused_path = config.output_dir / "original_diffused.png"
    perturbed_diffused_path = config.output_dir / "perturbed_diffused.png"
    report_path = config.output_dir / "report.json"

    original_pil = load_pil_image(config.input_path)
    save_pil(original_path, original_pil)

    original_array = load_image(config.input_path)
    perturbed_array = apply_perturbation_pipeline(original_array, config.perturbations)
    save_image(perturbed_path, perturbed_array)

    diffusion_report = {**asdict(config.diffusion), "resolved_device": resolved_device, "execution_mode": "sequential"}
    perturbed_pil = load_pil_image(perturbed_path)
    if use_gpu_optimizations:
        try:
            original_diffused, perturbed_diffused = edit_images(
                [original_pil, perturbed_pil],
                config.prompt,
                config.diffusion,
            )
            diffusion_report["execution_mode"] = "batched"
        except Exception as exc:
            diffusion_report["execution_mode"] = "sequential_after_batch_error"
            diffusion_report["batch_error"] = f"{type(exc).__name__}: {exc}"
            original_diffused = edit_image(original_pil, config.prompt, config.diffusion)
            perturbed_diffused = edit_image(perturbed_pil, config.prompt, config.diffusion)
    else:
        original_diffused = edit_image(original_pil, config.prompt, config.diffusion)
        perturbed_diffused = edit_image(perturbed_pil, config.prompt, config.diffusion)
    save_pil(original_diffused_path, original_diffused)
    save_pil(perturbed_diffused_path, perturbed_diffused)

    deepface_report = None
    if config.deepface.enabled:
        deepface_report = compare_images(
            original_diffused_path,
            perturbed_diffused_path,
            config.deepface,
            allow_parallel=use_gpu_optimizations,
        )

    report = {
        "config_path": str(config.config_path),
        "input": str(config.input_path),
        "output_dir": str(config.output_dir),
        "prompt": config.prompt,
        "seed": config.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "deepface": deepface_report,
        "perturbations": [asdict(step) for step in config.perturbations],
        "diffusion": diffusion_report,
        "outputs": {
            "original": str(original_path),
            "perturbed": str(perturbed_path),
            "original_diffused": str(original_diffused_path),
            "perturbed_diffused": str(perturbed_diffused_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the geometric-v1 full pipeline")
    parser.add_argument("--config", type=Path, default=Path("pipeline.json"))
    args = parser.parse_args(argv)
    report = run_pipeline(args.config)
    print(json.dumps(report["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
