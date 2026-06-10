from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_DEEPFACE_MODELS, DeepFaceConfig, load_pipeline_config
from .events import EventCallback, StopCallback, emit_event, is_stop_requested, with_event_context
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
    deepface_worker: dict[str, Any]
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
    deepface_worker = data.get("deepface_worker", {})
    if not isinstance(deepface_worker, dict):
        raise ValueError("deepface_worker must be an object when provided")

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
        deepface_worker={
            "enabled": bool(deepface_worker.get("enabled", True)),
            "max_attempts_per_worker": int(deepface_worker.get("max_attempts_per_worker", 100)),
            "timeout_seconds": float(deepface_worker.get("timeout_seconds", 600.0)),
            "restart_on_failure": bool(deepface_worker.get("restart_on_failure", True)),
        },
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
    models = diffusion.get("models")
    if isinstance(models, dict):
        for model_name in ("instruct_pix2pix", "flux2_klein"):
            if isinstance(models.get(model_name), dict):
                models[model_name]["seed"] = attempt_seed

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


def _diffusion_report_from_pipeline_config(pipeline_config: Path) -> dict[str, str]:
    diffusion = load_pipeline_config(pipeline_config).diffusion
    return {
        "selected_model": diffusion.selected_model,
        "selected_model_id": diffusion.selected_model_id,
        "used_model": diffusion.selected_model,
        "used_model_id": diffusion.selected_model_id,
    }


def _diffusion_report_from_pipeline_report(report: dict[str, Any]) -> dict[str, Any] | None:
    diffusion = report.get("diffusion")
    if not isinstance(diffusion, dict):
        return None
    return {
        "selected_model": diffusion.get("selected_model") or diffusion.get("used_model"),
        "selected_model_id": diffusion.get("selected_model_id") or diffusion.get("used_model_id"),
        "used_model": diffusion.get("used_model") or diffusion.get("selected_model"),
        "used_model_id": diffusion.get("used_model_id") or diffusion.get("selected_model_id"),
    }


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


class _PersistentDeepFaceWorker:
    def __init__(self, worker_config: dict[str, Any]) -> None:
        self.max_attempts = max(1, int(worker_config.get("max_attempts_per_worker", 100)))
        self.timeout_seconds = max(1.0, float(worker_config.get("timeout_seconds", 600.0)))
        self.restart_on_failure = bool(worker_config.get("restart_on_failure", True))
        self.process: subprocess.Popen[str] | None = None
        self.stdout_queue: queue.Queue[str | None] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=80)
        self.attempts_in_process = 0
        atexit.register(self.close)

    def _reader(self, stream, target_queue: queue.Queue[str | None] | None = None) -> None:
        try:
            for line in stream:
                if target_queue is None:
                    self.stderr_tail.append(line.rstrip())
                else:
                    target_queue.put(line)
        finally:
            if target_queue is not None:
                target_queue.put(None)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        return env

    def start(self) -> None:
        self.close()
        self.stdout_queue = queue.Queue()
        self.stderr_tail = deque(maxlen=80)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "geometric_v1.deepface_worker"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        threading.Thread(target=self._reader, args=(self.process.stdout, self.stdout_queue), daemon=True).start()
        threading.Thread(target=self._reader, args=(self.process.stderr, None), daemon=True).start()
        self.attempts_in_process = 0

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            with suppress(Exception):
                assert process.stdin is not None
                request_id = f"shutdown-{uuid.uuid4().hex}"
                process.stdin.write(json.dumps({"id": request_id, "type": "shutdown"}) + "\n")
                process.stdin.flush()
            with suppress(Exception):
                process.wait(timeout=5)
        if process.poll() is None:
            with suppress(Exception):
                process.terminate()
            with suppress(Exception):
                process.wait(timeout=5)
        if process.poll() is None:
            with suppress(Exception):
                process.kill()

    def _stderr_summary(self) -> str:
        return "\n".join(self.stderr_tail).strip()

    def _read_response(self, request_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            process = self.process
            if process is None:
                raise RuntimeError("DeepFace worker is not running")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"DeepFace worker timed out after {self.timeout_seconds:.1f}s")
            if process.poll() is not None and self.stdout_queue.empty():
                detail = self._stderr_summary()
                raise RuntimeError(f"DeepFace worker exited with code {process.returncode}: {detail}")
            try:
                line = self.stdout_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if line is None:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self.stderr_tail.append(f"non-json stdout: {line.rstrip()}")
                continue
            if response.get("id") == request_id:
                return response

    def compare(self, image_a: Path, image_b: Path, sampled_config_path: Path) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = 2 if self.restart_on_failure else 1
        for retry_index in range(attempts):
            if self.process is None or self.process.poll() is not None or self.attempts_in_process >= self.max_attempts:
                self.start()
            request_id = f"deepface-{uuid.uuid4().hex}"
            request = {
                "id": request_id,
                "image_a": str(image_a),
                "image_b": str(image_b),
                "config": str(sampled_config_path),
            }
            try:
                assert self.process is not None and self.process.stdin is not None
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                response = self._read_response(request_id)
                self.attempts_in_process += 1
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error", "DeepFace worker failed")))
                result = response["result"]
                execution = result.setdefault("execution", {})
                if isinstance(execution, dict):
                    execution["persistent_worker"] = True
                    execution["worker_reused_attempts"] = self.attempts_in_process
                    execution["max_attempts_per_worker"] = self.max_attempts
                return result
            except Exception as exc:
                last_error = exc
                self.close()
                if retry_index + 1 >= attempts:
                    break
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _deepface_error_report(
    image_a: Path,
    image_b: Path,
    config: DeepFaceConfig,
    error: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for model_name in DEFAULT_DEEPFACE_MODELS:
        enabled = bool(config.models.get(model_name, False))
        if enabled:
            models[model_name] = {
                "enabled": True,
                "ok": False,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
            }
        else:
            models[model_name] = {"enabled": False, "skipped": True}
    return {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "detector_backend": config.detector_backend,
        "distance_metric": config.distance_metric,
        "execution": {
            "isolated_subprocess": True,
            "parallel": False,
            "resolved_workers": 1,
            "error": error,
        },
        "models": models,
        "summary": {
            "successful_models": 0,
            "mean_match_percent": None,
            "min_match_percent": None,
            "max_match_percent": None,
        },
    }


def _emit_deepface_report_events(
    event_callback: EventCallback | None,
    event_context: dict[str, Any],
    deepface_report: dict[str, Any],
) -> None:
    model_results = deepface_report.get("models", {})
    if not isinstance(model_results, dict):
        return
    ok_values: list[float] = []
    for model_name, model_result in model_results.items():
        if not isinstance(model_result, dict):
            continue
        if not model_result.get("enabled"):
            emit_event(event_callback, "deepface_model_skipped", **event_context, model=model_name, status="skipped")
            continue
        if model_result.get("ok") and model_result.get("match_percent") is not None:
            percentage = float(model_result["match_percent"])
            ok_values.append(percentage)
            emit_event(
                event_callback,
                "deepface_model_completed",
                **event_context,
                model=model_name,
                percentage=percentage,
                verified=bool(model_result.get("verified")),
                status="completed",
                elapsed_seconds=model_result.get("elapsed_seconds"),
            )
            emit_event(
                event_callback,
                "running_mean_updated",
                **event_context,
                completed_models=len(ok_values),
                mean_match_percent=sum(ok_values) / len(ok_values),
                min_match_percent=min(ok_values),
                max_match_percent=max(ok_values),
            )
        else:
            emit_event(
                event_callback,
                "deepface_model_error",
                **event_context,
                model=model_name,
                status="error",
                error=model_result.get("error", "DeepFace subprocess failed"),
                elapsed_seconds=model_result.get("elapsed_seconds"),
            )


def _run_deepface_isolated(
    report: dict[str, Any],
    sampled_config_path: Path,
    event_callback: EventCallback | None,
    event_context: dict[str, Any],
    worker: _PersistentDeepFaceWorker | None = None,
) -> dict[str, Any]:
    outputs = report.get("outputs", {})
    image_a = Path(str(outputs.get("original_diffused", "")))
    image_b = Path(str(outputs.get("perturbed_diffused", "")))
    deepface_config = load_pipeline_config(sampled_config_path).deepface
    enabled_models = [
        model_name
        for model_name, enabled in deepface_config.models.items()
        if model_name in DEFAULT_DEEPFACE_MODELS and enabled
    ]
    for model_name in DEFAULT_DEEPFACE_MODELS:
        if model_name in enabled_models:
            emit_event(event_callback, "deepface_model_pending", **event_context, model=model_name, status="pending")
        else:
            emit_event(event_callback, "deepface_model_skipped", **event_context, model=model_name, status="skipped")
    for model_name in enabled_models:
        emit_event(event_callback, "deepface_model_running", **event_context, model=model_name, status="running")

    started = time.perf_counter()
    output_path = Path(str(report.get("output_dir", sampled_config_path.parent))) / "deepface_report.json"
    if worker is not None:
        try:
            deepface_report = worker.compare(image_a, image_b, sampled_config_path)
        except Exception as exc:
            error = f"DeepFace worker failed: {type(exc).__name__}: {exc}"
            deepface_report = _deepface_error_report(
                image_a,
                image_b,
                deepface_config,
                error,
                time.perf_counter() - started,
            )
        _write_json(output_path, deepface_report)
        _emit_deepface_report_events(event_callback, event_context, deepface_report)
        return deepface_report

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    command = [
        sys.executable,
        "-m",
        "geometric_v1.deepface_cli",
        "--image-a",
        str(image_a),
        "--image-b",
        str(image_b),
        "--output",
        str(output_path),
        "--config",
        str(sampled_config_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "no subprocess output"
        error = f"DeepFace subprocess exited with code {completed.returncode}: {detail}"
        deepface_report = _deepface_error_report(image_a, image_b, deepface_config, error, elapsed)
        _write_json(output_path, deepface_report)
        _emit_deepface_report_events(event_callback, event_context, deepface_report)
        return deepface_report

    try:
        deepface_report = _read_json(output_path)
    except Exception as exc:
        error = f"DeepFace subprocess did not write a valid report: {type(exc).__name__}: {exc}"
        deepface_report = _deepface_error_report(image_a, image_b, deepface_config, error, elapsed)
        _write_json(output_path, deepface_report)
    execution = deepface_report.setdefault("execution", {})
    if isinstance(execution, dict):
        execution["isolated_subprocess"] = True
        execution["parallel"] = False
        execution["resolved_workers"] = 1
    _emit_deepface_report_events(event_callback, event_context, deepface_report)
    return deepface_report


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
        "diffusion": _diffusion_report_from_pipeline_report(report),
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
    diffusion_report: dict[str, str],
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
        "diffusion": diffusion_report,
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


def _emit_brute_score_bounds(
    event_callback: EventCallback | None,
    score_values: list[tuple[int, float, str, str]],
) -> None:
    if not score_values:
        return
    minimum = min(score_values, key=lambda item: item[1])
    maximum = max(score_values, key=lambda item: item[1])
    emit_event(
        event_callback,
        "min_max_score_updated",
        min_score={
            "run_number": minimum[0],
            "mean_percentage": minimum[1],
            "status": minimum[2],
            "path": minimum[3],
            "report": str(Path(minimum[3]) / "report.json"),
        },
        max_score={
            "run_number": maximum[0],
            "mean_percentage": maximum[1],
            "status": maximum[2],
            "path": maximum[3],
            "report": str(Path(maximum[3]) / "report.json"),
        },
    )


def run_brute_force(
    config_path: Path,
    attempt_seeds: list[int] | None = None,
    rng_seed: int | None = None,
    event_callback: EventCallback | None = None,
    stop_requested: StopCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    emit_event(event_callback, "run_started", run_type="brute", config_path=str(config_path))
    config = load_brute_config(config_path)
    if attempt_seeds is not None and len(attempt_seeds) != config.trials:
        raise ValueError("attempt_seeds length must match brute trials")
    pipeline_data = _read_json(config.pipeline_config)
    diffusion_report = _diffusion_report_from_pipeline_config(config.pipeline_config)
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
        "deepface_worker": config.deepface_worker,
        "diffusion": diffusion_report,
        "attempts": [],
    }
    stopped = False
    score_values: list[tuple[int, float, str, str]] = []
    pipeline_deepface_enabled = load_pipeline_config(config.pipeline_config).deepface.enabled
    deepface_worker = (
        _PersistentDeepFaceWorker(config.deepface_worker)
        if pipeline_deepface_enabled and bool(config.deepface_worker.get("enabled", True))
        else None
    )

    for attempt in range(config.trials):
        if is_stop_requested(stop_requested):
            stopped = True
            emit_event(
                event_callback,
                "run_stopping",
                run_type="brute",
                reason="stop requested before next attempt",
                next_attempt=attempt,
            )
            break

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
                if record.get("average_match_percent") is not None:
                    score_values.append(
                        (attempt, float(record["average_match_percent"]), str(record["status"]), str(record["folder"]))
                    )
                    _emit_brute_score_bounds(event_callback, score_values)
                emit_event(
                    event_callback,
                    "brute_attempt_skipped",
                    run_number=attempt,
                    run_name=run_name,
                    status=record["status"],
                    action="skipped",
                    folder=record["folder"],
                    summary=brute_report["summary"],
                )
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
        emit_event(
            event_callback,
            "brute_attempt_started",
            run_number=attempt,
            run_name=run_name,
            attempt_seed=attempt_seed,
            action=action,
            sampled_config=str(sampled_config_path),
            sampled_perturbations=sampled_config.get("perturbations", []),
        )

        pipeline_error: str | None = None
        try:
            report = run_pipeline(
                sampled_config_path,
                event_callback=with_event_context(event_callback, run_number=attempt, run_name=run_name),
                run_deepface=False,
            )
            if load_pipeline_config(sampled_config_path).deepface.enabled:
                report["deepface"] = _run_deepface_isolated(
                    report,
                    sampled_config_path,
                    event_callback,
                    {"stage": "pipeline", "run_number": attempt, "run_name": run_name},
                    worker=deepface_worker,
                )
        except Exception as exc:  # Keep going so one bad sample does not stop the search.
            pipeline_error = f"{type(exc).__name__}: {exc}"
            report = _error_report(
                config,
                sampled_config_path,
                staging_dir,
                attempt,
                attempt_seed,
                diffusion_report,
                exc,
            )
            emit_event(
                event_callback,
                "brute_attempt_failed",
                run_number=attempt,
                run_name=run_name,
                attempt_seed=attempt_seed,
                error=pipeline_error,
            )

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
            "diffusion": _diffusion_report_from_pipeline_report(report),
            "saved": True,
            "folder": str(final_dir),
            "action": action,
            "archived_incomplete": archived_incomplete,
        }
        brute_report["attempts"].append(attempt_record)
        brute_report["summary"] = _summarize_attempts(brute_report["attempts"])
        _write_json(brute_report_path, brute_report)

        if average is not None:
            score_values.append((attempt, float(average), status, str(final_dir)))
            _emit_brute_score_bounds(event_callback, score_values)
        emit_event(
            event_callback,
            "brute_attempt_completed",
            run_number=attempt,
            run_name=run_name,
            attempt_seed=attempt_seed,
            status=status,
            action=action,
            average_match_percent=average,
            counted_models=counted_models,
            errored_models=errored_models,
            folder=str(final_dir),
            summary=brute_report["summary"],
        )

        average_text = "none" if average is None else f"{average:.4f}"
        print(f"{run_name}: average={average_text} status={status} action={action}")

    brute_report["status"] = "stopped" if stopped else "completed"
    brute_report["elapsed_seconds"] = time.perf_counter() - started
    brute_report["summary"] = _summarize_attempts(brute_report["attempts"])
    _write_json(brute_report_path, brute_report)
    if deepface_worker is not None:
        deepface_worker.close()
    with suppress(OSError):
        working_dir.rmdir()
    emit_event(
        event_callback,
        "run_completed" if not stopped else "run_stopped",
        run_type="brute",
        status=brute_report["status"],
        report_path=str(brute_report_path),
        summary=brute_report["summary"],
    )
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
