from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import run_pipeline


INTEGER_FIELDS = {"grid", "coefficients"}


@dataclass(frozen=True)
class BruteConfig:
    config_path: Path
    pipeline_config: Path
    output_dir: Path
    trials: int
    success_threshold: float
    seed: int
    randomize_attempt_seed: bool
    attempt_seed_range: tuple[int, int]
    save_unsuccessful: bool
    ranges: dict[str, dict[str, list[float | int]]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_brute_config(path: Path) -> BruteConfig:
    path = path.resolve()
    data = _read_json(path)
    base_dir = path.parent
    attempt_range = data.get("attempt_seed_range", [1, 2_147_483_647])
    if not isinstance(attempt_range, list) or len(attempt_range) != 2:
        raise ValueError("attempt_seed_range must be a two-item list")
    min_seed = int(attempt_range[0])
    max_seed = int(attempt_range[1])
    if min_seed > max_seed:
        raise ValueError("attempt_seed_range minimum cannot be greater than maximum")

    ranges = data.get("ranges", {})
    if not isinstance(ranges, dict):
        raise ValueError("ranges must be an object")

    return BruteConfig(
        config_path=path,
        pipeline_config=_resolve(base_dir, str(data.get("pipeline_config", "pipeline.json"))).resolve(),
        output_dir=_resolve(base_dir, str(data.get("output_dir", "output/brute_force"))).resolve(),
        trials=int(data.get("trials", 1)),
        success_threshold=float(data.get("success_threshold", 50.0)),
        seed=int(data.get("seed", 42)),
        randomize_attempt_seed=bool(data.get("randomize_attempt_seed", True)),
        attempt_seed_range=(min_seed, max_seed),
        save_unsuccessful=bool(data.get("save_unsuccessful", True)),
        ranges=ranges,
    )


def _sample_parameter(rng: random.Random, field: str, bounds: list[float | int]) -> float | int:
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"range for {field} must be a two-item list")
    lower = bounds[0]
    upper = bounds[1]
    if field in INTEGER_FIELDS:
        low_int = int(lower)
        high_int = int(upper)
        if low_int > high_int:
            raise ValueError(f"range for {field} has minimum greater than maximum")
        return rng.randint(low_int, high_int)

    low_float = float(lower)
    high_float = float(upper)
    if low_float > high_float:
        raise ValueError(f"range for {field} has minimum greater than maximum")
    return rng.uniform(low_float, high_float)


def _next_attempt_seed(config: BruteConfig, rng: random.Random) -> int:
    if not config.randomize_attempt_seed:
        return config.seed
    return rng.randint(config.attempt_seed_range[0], config.attempt_seed_range[1])


def _make_sampled_pipeline(
    pipeline_data: dict[str, Any],
    pipeline_path: Path,
    output_dir: Path,
    attempt_seed: int,
    rng: random.Random,
    ranges: dict[str, dict[str, list[float | int]]],
) -> dict[str, Any]:
    sampled = deepcopy(pipeline_data)
    sampled["seed"] = attempt_seed
    sampled["input"] = str(_resolve(pipeline_path.parent, str(pipeline_data["input"])).resolve())
    sampled["output_dir"] = str(output_dir.resolve())

    diffusion = sampled.setdefault("diffusion", {})
    diffusion["seed"] = attempt_seed

    enabled_index = 0
    for step in sampled.get("perturbations", []):
        if not bool(step.get("enabled", True)):
            continue
        method = str(step["method"])
        step["seed"] = attempt_seed + enabled_index
        enabled_index += 1
        for field, bounds in ranges.get(method, {}).items():
            step[field] = _sample_parameter(rng, field, bounds)
    return sampled


def _deepface_score(report: dict[str, Any]) -> tuple[float | None, int, list[str]]:
    deepface = report.get("deepface")
    if not isinstance(deepface, dict):
        return None, 0, []

    values: list[float] = []
    errored_models: list[str] = []
    for model_name, model_report in deepface.get("models", {}).items():
        if not model_report.get("enabled"):
            continue
        if model_report.get("ok") and model_report.get("match_percent") is not None:
            values.append(float(model_report["match_percent"]))
        elif model_report.get("ok") is False:
            errored_models.append(str(model_name))

    if not values:
        return None, 0, errored_models
    return sum(values) / len(values), len(values), errored_models


def _update_final_paths(report: dict[str, Any], sampled_config: dict[str, Any], final_dir: Path) -> None:
    sampled_config["output_dir"] = str(final_dir)
    report["config_path"] = str(final_dir / "sampled_config.json")
    report["output_dir"] = str(final_dir)
    outputs = report.setdefault("outputs", {})
    for name, filename in {
        "original": "original.png",
        "perturbed": "perturbed.png",
        "original_diffused": "original_diffused.png",
        "perturbed_diffused": "perturbed_diffused.png",
        "report": "report.json",
    }.items():
        candidate = final_dir / filename
        if name == "report" or candidate.exists():
            outputs[name] = str(candidate)
        else:
            outputs.pop(name, None)

    deepface = report.get("deepface")
    if isinstance(deepface, dict):
        for key, filename in {"image_a": "original_diffused.png", "image_b": "perturbed_diffused.png"}.items():
            candidate = final_dir / filename
            if candidate.exists():
                deepface[key] = str(candidate)
            else:
                deepface.pop(key, None)


def _preflight_output_dirs(config: BruteConfig) -> None:
    if config.trials < 1:
        raise ValueError("trials must be at least 1")
    for attempt in range(config.trials):
        run_name = f"run_{attempt:06d}"
        for bucket in ("successful", "unsuccessful", "failures"):
            candidate = config.output_dir / bucket / run_name
            if candidate.exists():
                raise FileExistsError(f"refusing to overwrite existing run folder: {candidate}")


def _error_report(
    config: BruteConfig,
    sampled_config_path: Path,
    staging_dir: Path,
    attempt: int,
    attempt_seed: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "config_path": str(sampled_config_path),
        "output_dir": str(staging_dir),
        "seed": attempt_seed,
        "outputs": {
            "report": str(staging_dir / "report.json"),
        },
        "deepface": None,
        "brute_force": {
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "success_threshold": config.success_threshold,
            "success": False,
            "average_match_percent": None,
            "counted_models": 0,
            "errored_models": [],
            "pipeline_error": f"{type(exc).__name__}: {exc}",
        },
    }


def run_brute_force(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_brute_config(config_path)
    pipeline_data = _read_json(config.pipeline_config)
    rng = random.Random(config.seed)

    successful_dir = config.output_dir / "successful"
    unsuccessful_dir = config.output_dir / "unsuccessful"
    failures_dir = config.output_dir / "failures"
    working_dir = config.output_dir / "_working"
    successful_dir.mkdir(parents=True, exist_ok=True)
    unsuccessful_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    _preflight_output_dirs(config)

    brute_report_path = config.output_dir / "brute_report.json"
    brute_report: dict[str, Any] = {
        "config_path": str(config.config_path),
        "pipeline_config": str(config.pipeline_config),
        "output_dir": str(config.output_dir),
        "trials": config.trials,
        "success_threshold": config.success_threshold,
        "seed": config.seed,
        "randomize_attempt_seed": config.randomize_attempt_seed,
        "attempt_seed_range": list(config.attempt_seed_range),
        "save_unsuccessful": config.save_unsuccessful,
        "attempts": [],
    }

    for attempt in range(config.trials):
        run_name = f"run_{attempt:06d}"
        attempt_seed = _next_attempt_seed(config, rng)
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{run_name}_", dir=working_dir))
        sampled_config_path = staging_dir / "sampled_config.json"
        sampled_config = _make_sampled_pipeline(
            pipeline_data=pipeline_data,
            pipeline_path=config.pipeline_config,
            output_dir=staging_dir,
            attempt_seed=attempt_seed,
            rng=rng,
            ranges=config.ranges,
        )
        _write_json(sampled_config_path, sampled_config)

        pipeline_error: str | None = None
        try:
            report = run_pipeline(sampled_config_path)
        except Exception as exc:  # Keep going so one bad sample does not stop the search.
            pipeline_error = f"{type(exc).__name__}: {exc}"
            report = _error_report(config, sampled_config_path, staging_dir, attempt, attempt_seed, exc)

        average, counted_models, errored_models = _deepface_score(report)
        has_error = pipeline_error is not None or bool(errored_models) or average is None
        success = not has_error and average <= config.success_threshold
        status = "failure" if has_error else ("successful" if success else "unsuccessful")
        bucket_dir = {"successful": successful_dir, "unsuccessful": unsuccessful_dir, "failure": failures_dir}[status]
        final_dir = bucket_dir / run_name

        report["brute_force"] = {
            "attempt": attempt,
            "attempt_seed": attempt_seed,
            "success_threshold": config.success_threshold,
            "status": status,
            "success": success,
            "average_match_percent": average,
            "counted_models": counted_models,
            "errored_models": errored_models,
            "pipeline_error": pipeline_error,
        }

        if final_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run folder: {final_dir}")
        shutil.move(str(staging_dir), str(final_dir))
        if status == "failure" and not config.save_unsuccessful:
            for filename in ("original.png", "perturbed.png", "original_diffused.png", "perturbed_diffused.png"):
                with suppress(OSError):
                    (final_dir / filename).unlink()
        _update_final_paths(report, sampled_config, final_dir)
        _write_json(final_dir / "sampled_config.json", sampled_config)
        _write_json(final_dir / "report.json", report)

        attempt_record = {
            "attempt": attempt,
            "run_name": run_name,
            "attempt_seed": attempt_seed,
            "status": status,
            "success": success,
            "average_match_percent": average,
            "counted_models": counted_models,
            "errored_models": errored_models,
            "pipeline_error": pipeline_error,
            "saved": True,
            "folder": str(final_dir),
        }
        brute_report["attempts"].append(attempt_record)
        _write_json(brute_report_path, brute_report)

        average_text = "none" if average is None else f"{average:.4f}"
        print(f"{run_name}: average={average_text} status={status}")

    brute_report["elapsed_seconds"] = time.perf_counter() - started
    brute_report["summary"] = {
        "successful": sum(1 for item in brute_report["attempts"] if item["status"] == "successful"),
        "unsuccessful": sum(1 for item in brute_report["attempts"] if item["status"] == "unsuccessful"),
        "failures": sum(1 for item in brute_report["attempts"] if item["status"] == "failure"),
        "saved": sum(1 for item in brute_report["attempts"] if item["saved"]),
    }
    _write_json(brute_report_path, brute_report)
    with suppress(OSError):
        working_dir.rmdir()
    return brute_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run random brute-force perturbation search")
    parser.add_argument("--config", type=Path, default=Path("brute.json"))
    args = parser.parse_args(argv)
    report = run_brute_force(args.config)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
