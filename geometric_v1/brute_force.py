from __future__ import annotations

import argparse
import hashlib
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
RUN_BUCKETS = {
    "successful": "successful",
    "unsuccessful": "unsuccessful",
    "failure": "failures",
}


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
    resume: bool
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
        resume=bool(data.get("resume", False)),
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


def _attempt_rng(base_seed: int, attempt: int) -> random.Random:
    digest = hashlib.sha256(f"geometric-v1:{base_seed}:{attempt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _planned_attempt_seeds(
    config: BruteConfig,
    explicit_attempt_seeds: list[int] | None,
    stored_attempt_seeds: dict[int, int],
) -> list[int]:
    if explicit_attempt_seeds is not None:
        seeds = list(explicit_attempt_seeds)
    else:
        rng = random.Random(config.seed)
        seeds = [_next_attempt_seed(config, rng) for _ in range(config.trials)]

    for attempt, stored_seed in stored_attempt_seeds.items():
        if 0 <= attempt < len(seeds):
            seeds[attempt] = stored_seed
    return seeds


def _load_existing_brute_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _stored_attempt_seeds(report: dict[str, Any] | None) -> dict[int, int]:
    if not report:
        return {}
    seeds: dict[int, int] = {}
    for item in report.get("attempts", []):
        try:
            attempt = int(item["attempt"])
            attempt_seed = int(item["attempt_seed"])
        except (KeyError, TypeError, ValueError):
            continue
        seeds[attempt] = attempt_seed
    return seeds


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


def _run_name(attempt: int) -> str:
    return f"run_{attempt:06d}"


def _run_folder(config: BruteConfig, status: str, attempt: int) -> Path:
    return config.output_dir / RUN_BUCKETS[status] / _run_name(attempt)


def _completed_attempt_record(folder: Path, attempt: int) -> dict[str, Any] | None:
    report_path = folder / "report.json"
    sampled_path = folder / "sampled_config.json"
    if not report_path.exists() or not sampled_path.exists():
        return None
    try:
        report = _read_json(report_path)
    except Exception:
        return None
    brute_force = report.get("brute_force")
    if not isinstance(brute_force, dict) or not brute_force.get("status"):
        return None

    status = str(brute_force.get("status"))
    return {
        "attempt": int(brute_force.get("attempt", attempt)),
        "run_name": folder.name,
        "attempt_seed": brute_force.get("attempt_seed"),
        "status": status,
        "success": bool(brute_force.get("success", status == "successful")),
        "average_match_percent": brute_force.get("average_match_percent"),
        "counted_models": int(brute_force.get("counted_models", 0) or 0),
        "errored_models": brute_force.get("errored_models", []),
        "pipeline_error": brute_force.get("pipeline_error"),
        "saved": True,
        "folder": str(folder),
        "action": "skipped",
    }


def _find_attempt_outputs(config: BruteConfig, attempt: int) -> tuple[list[dict[str, Any]], list[Path]]:
    completed: list[dict[str, Any]] = []
    incomplete: list[Path] = []
    for status in RUN_BUCKETS:
        folder = _run_folder(config, status, attempt)
        if not folder.exists():
            continue
        record = _completed_attempt_record(folder, attempt)
        if record is None:
            incomplete.append(folder)
        else:
            completed.append(record)
    return completed, incomplete


def _archive_incomplete_run(config: BruteConfig, path: Path, attempt: int) -> str:
    failures_dir = config.output_dir / RUN_BUCKETS["failure"]
    failures_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"incomplete_{path.parent.name}_{_run_name(attempt)}_{stamp}"
    destination = failures_dir / base_name
    counter = 1
    while destination.exists():
        destination = failures_dir / f"{base_name}_{counter}"
        counter += 1
    shutil.move(str(path), str(destination))
    return str(destination)


def _clean_working_attempts(working_dir: Path, attempt: int) -> list[str]:
    if not working_dir.exists():
        return []
    removed: list[str] = []
    prefix = f"{_run_name(attempt)}_"
    for child in working_dir.iterdir():
        if child.name.startswith(prefix):
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
    return removed


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


def _summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "successful": sum(1 for item in attempts if item.get("status") == "successful"),
        "unsuccessful": sum(1 for item in attempts if item.get("status") == "unsuccessful"),
        "failures": sum(1 for item in attempts if item.get("status") == "failure"),
        "saved": sum(1 for item in attempts if item.get("saved")),
        "completed_runs": sum(1 for item in attempts if item.get("status") in RUN_BUCKETS),
        "skipped_runs": sum(1 for item in attempts if item.get("action") == "skipped"),
        "resumed_runs": sum(1 for item in attempts if item.get("action") == "resumed"),
        "executed_runs": sum(1 for item in attempts if item.get("action") in {"executed", "resumed"}),
    }


def run_brute_force(
    config_path: Path,
    attempt_seeds: list[int] | None = None,
    rng_seed: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_brute_config(config_path)
    if attempt_seeds is not None and len(attempt_seeds) != config.trials:
        raise ValueError("attempt_seeds length must match brute trials")
    pipeline_data = _read_json(config.pipeline_config)
    parameter_seed = config.seed if rng_seed is None else rng_seed

    successful_dir = config.output_dir / "successful"
    unsuccessful_dir = config.output_dir / "unsuccessful"
    failures_dir = config.output_dir / "failures"
    working_dir = config.output_dir / "_working"
    successful_dir.mkdir(parents=True, exist_ok=True)
    unsuccessful_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    if not config.resume:
        _preflight_output_dirs(config)

    brute_report_path = config.output_dir / "brute_report.json"
    existing_brute_report = _load_existing_brute_report(brute_report_path)
    stored_seeds = _stored_attempt_seeds(existing_brute_report)
    planned_attempt_seeds = _planned_attempt_seeds(config, attempt_seeds, stored_seeds)
    had_existing_state = existing_brute_report is not None or any(
        _run_folder(config, status, attempt).exists()
        for attempt in range(config.trials)
        for status in RUN_BUCKETS
    )
    brute_report: dict[str, Any] = {
        "config_path": str(config.config_path),
        "pipeline_config": str(config.pipeline_config),
        "output_dir": str(config.output_dir),
        "trials": config.trials,
        "success_threshold": config.success_threshold,
        "seed": config.seed,
        "randomize_attempt_seed": config.randomize_attempt_seed,
        "attempt_seed_range": list(config.attempt_seed_range),
        "resume": config.resume,
        "save_unsuccessful": config.save_unsuccessful,
        "attempts": [],
    }

    for attempt in range(config.trials):
        run_name = _run_name(attempt)
        attempt_seed = planned_attempt_seeds[attempt]
        completed_records, incomplete_paths = _find_attempt_outputs(config, attempt)
        archived_incomplete: list[str] = []

        if len(completed_records) > 1:
            folders = ", ".join(record["folder"] for record in completed_records)
            raise FileExistsError(f"multiple completed folders found for {run_name}: {folders}")

        if completed_records:
            if config.resume:
                for path in incomplete_paths:
                    archived_incomplete.append(_archive_incomplete_run(config, path, attempt))
                archived_incomplete.extend(_clean_working_attempts(working_dir, attempt))
                record = completed_records[0]
                if record.get("attempt_seed") is None:
                    record["attempt_seed"] = attempt_seed
                record["archived_incomplete"] = archived_incomplete
                brute_report["attempts"].append(record)
                brute_report["summary"] = _summarize_attempts(brute_report["attempts"])
                _write_json(brute_report_path, brute_report)
                print(f"{run_name}: skipped completed {record['status']}")
                continue

        if incomplete_paths and not config.resume:
            folders = ", ".join(str(path) for path in incomplete_paths)
            raise FileExistsError(f"refusing to overwrite incomplete run folder(s): {folders}")

        if config.resume:
            for path in incomplete_paths:
                archived_incomplete.append(_archive_incomplete_run(config, path, attempt))
            archived_incomplete.extend(_clean_working_attempts(working_dir, attempt))

        action = "resumed" if config.resume and had_existing_state else "executed"
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{run_name}_", dir=working_dir))
        sampled_config_path = staging_dir / "sampled_config.json"
        sampled_config = _make_sampled_pipeline(
            pipeline_data=pipeline_data,
            pipeline_path=config.pipeline_config,
            output_dir=staging_dir,
            attempt_seed=attempt_seed,
            rng=_attempt_rng(parameter_seed, attempt),
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
            "action": action,
            "archived_incomplete": archived_incomplete,
        }
        brute_report["attempts"].append(attempt_record)
        brute_report["summary"] = _summarize_attempts(brute_report["attempts"])
        _write_json(brute_report_path, brute_report)

        average_text = "none" if average is None else f"{average:.4f}"
        print(f"{run_name}: average={average_text} status={status} action={action}")

    brute_report["elapsed_seconds"] = time.perf_counter() - started
    brute_report["summary"] = _summarize_attempts(brute_report["attempts"])
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
