from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brute_force import load_brute_config, run_brute_force


DEFAULT_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


@dataclass(frozen=True)
class BatchBruteConfig:
    config_path: Path
    brute_config: Path
    pipeline_config: Path | None
    images_dir: Path
    image_extensions: list[str]
    recursive: bool
    prompts: list[str]
    output_dir: Path
    skip_existing: bool
    parallel_combinations: int


@dataclass(frozen=True)
class ComboPlan:
    image_index: int
    prompt_index: int
    image_path: Path
    prompt: str
    combo_dir: Path
    brute_config_path: Path
    attempt_seeds: list[int] | None
    rng_seed: int | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _sanitize(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return (cleaned or fallback)[:80]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def _normal_extensions(values: list[str] | None) -> list[str]:
    extensions = values or DEFAULT_IMAGE_EXTENSIONS
    normalized = []
    for extension in extensions:
        text = str(extension).lower()
        normalized.append(text if text.startswith(".") else f".{text}")
    return normalized


def load_batch_brute_config(path: Path) -> BatchBruteConfig:
    path = path.resolve()
    data = _read_json(path)
    base_dir = path.parent
    pipeline_value = data.get("pipeline_config")
    prompts = [str(prompt) for prompt in data.get("prompts", [])]
    if not prompts:
        raise ValueError("batch_brute config must include at least one prompt")

    return BatchBruteConfig(
        config_path=path,
        brute_config=_resolve(base_dir, str(data.get("brute_config", "brute.json"))).resolve(),
        pipeline_config=_resolve(base_dir, str(pipeline_value)).resolve() if pipeline_value else None,
        images_dir=_resolve(base_dir, str(data["images_dir"])).resolve(),
        image_extensions=_normal_extensions(data.get("image_extensions")),
        recursive=bool(data.get("recursive", False)),
        prompts=prompts,
        output_dir=_resolve(base_dir, str(data.get("output_dir", "output/batch_brute"))).resolve(),
        skip_existing=bool(data.get("skip_existing", True)),
        parallel_combinations=max(1, int(data.get("parallel_combinations", 1))),
    )


def _find_images(config: BatchBruteConfig) -> list[Path]:
    pattern = "**/*" if config.recursive else "*"
    images = [
        path.resolve()
        for path in config.images_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in config.image_extensions
    ]
    return sorted(images, key=lambda path: path.relative_to(config.images_dir).as_posix().lower())


def _unique_attempt_seeds(
    brute_seed: int,
    attempt_seed_range: tuple[int, int],
    total_attempts: int,
) -> list[int]:
    min_seed, max_seed = attempt_seed_range
    available = max_seed - min_seed + 1
    if total_attempts > available:
        raise ValueError("attempt_seed_range is too small to generate unique seeds for the whole batch")
    rng = random.Random(brute_seed)
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < total_attempts:
        value = rng.randint(min_seed, max_seed)
        if value in seen:
            continue
        seen.add(value)
        seeds.append(value)
    return seeds


def _combo_dirs(config: BatchBruteConfig, image_index: int, image_path: Path, prompt_index: int, prompt: str) -> Path:
    image_name = f"image_{image_index + 1:06d}_{_sanitize(image_path.stem, 'image')}"
    prompt_name = f"prompt_{prompt_index:06d}_{_prompt_hash(prompt)}"
    return config.output_dir / image_name / prompt_name


def _write_combo_configs(
    combo_dir: Path,
    image_path: Path,
    prompt: str,
    brute_data: dict[str, Any],
    pipeline_data: dict[str, Any],
) -> Path:
    combo_dir.mkdir(parents=True, exist_ok=True)
    combo_pipeline = deepcopy(pipeline_data)
    combo_pipeline["input"] = str(image_path)
    combo_pipeline["prompt"] = prompt
    combo_pipeline["output_dir"] = str(combo_dir)

    combo_pipeline_path = combo_dir / "combo_pipeline.json"
    _write_json(combo_pipeline_path, combo_pipeline)

    combo_brute = deepcopy(brute_data)
    combo_brute["pipeline_config"] = str(combo_pipeline_path)
    combo_brute["output_dir"] = str(combo_dir)
    combo_brute_path = combo_dir / "combo_brute.json"
    _write_json(combo_brute_path, combo_brute)
    return combo_brute_path


def _run_combo(plan: ComboPlan) -> dict[str, Any]:
    started = time.perf_counter()
    report = run_brute_force(plan.brute_config_path, attempt_seeds=plan.attempt_seeds, rng_seed=plan.rng_seed)
    summary = report.get("summary", {})
    return {
        "image_index": plan.image_index,
        "prompt_index": plan.prompt_index,
        "status": "completed",
        "image_path": str(plan.image_path),
        "prompt": plan.prompt,
        "output_dir": str(plan.combo_dir),
        "brute_report": str(plan.combo_dir / "brute_report.json"),
        "successful": int(summary.get("successful", 0)),
        "unsuccessful": int(summary.get("unsuccessful", 0)),
        "failures": int(summary.get("failures", 0)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _skipped_record(plan: ComboPlan) -> dict[str, Any]:
    brute_report_path = plan.combo_dir / "brute_report.json"
    successful = unsuccessful = failures = 0
    if brute_report_path.exists():
        try:
            summary = _read_json(brute_report_path).get("summary", {})
            successful = int(summary.get("successful", 0))
            unsuccessful = int(summary.get("unsuccessful", 0))
            failures = int(summary.get("failures", 0))
        except Exception:
            pass
    return {
        "image_index": plan.image_index,
        "prompt_index": plan.prompt_index,
        "status": "skipped",
        "image_path": str(plan.image_path),
        "prompt": plan.prompt,
        "output_dir": str(plan.combo_dir),
        "brute_report": str(brute_report_path),
        "successful": successful,
        "unsuccessful": unsuccessful,
        "failures": failures,
        "elapsed_seconds": 0.0,
    }


def _failure_record(plan: ComboPlan, exc: Exception) -> dict[str, Any]:
    return {
        "image_index": plan.image_index,
        "prompt_index": plan.prompt_index,
        "status": "failed",
        "image_path": str(plan.image_path),
        "prompt": plan.prompt,
        "output_dir": str(plan.combo_dir),
        "brute_report": str(plan.combo_dir / "brute_report.json"),
        "successful": 0,
        "unsuccessful": 0,
        "failures": 0,
        "error": f"{type(exc).__name__}: {exc}",
        "elapsed_seconds": 0.0,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completed": sum(1 for record in records if record["status"] == "completed"),
        "skipped": sum(1 for record in records if record["status"] == "skipped"),
        "failed": sum(1 for record in records if record["status"] == "failed"),
        "successful": sum(int(record.get("successful", 0)) for record in records),
        "unsuccessful": sum(int(record.get("unsuccessful", 0)) for record in records),
        "failures": sum(int(record.get("failures", 0)) for record in records),
    }


def run_batch_brute_force(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_batch_brute_config(config_path)
    brute_config = load_brute_config(config.brute_config)
    brute_data = _read_json(config.brute_config)
    pipeline_config_path = config.pipeline_config or brute_config.pipeline_config
    pipeline_data = _read_json(pipeline_config_path)
    images = _find_images(config)
    total_attempts = len(images) * len(config.prompts) * brute_config.trials

    config.output_dir.mkdir(parents=True, exist_ok=True)
    batch_report_path = config.output_dir / "batch_report.json"
    batch_report: dict[str, Any] = {
        "config_path": str(config.config_path),
        "brute_config": str(config.brute_config),
        "pipeline_config": str(pipeline_config_path),
        "images_dir": str(config.images_dir),
        "image_extensions": config.image_extensions,
        "recursive": config.recursive,
        "output_dir": str(config.output_dir),
        "skip_existing": config.skip_existing,
        "parallel_combinations": config.parallel_combinations,
        "total_images": len(images),
        "total_prompts": len(config.prompts),
        "total_planned_brute_attempts": total_attempts,
        "results": [],
    }
    _write_json(batch_report_path, batch_report)

    all_attempt_seeds: list[int] | None = None
    if brute_config.randomize_attempt_seed:
        all_attempt_seeds = _unique_attempt_seeds(brute_config.seed, brute_config.attempt_seed_range, total_attempts)

    sampling_seed_rng = random.Random(brute_config.seed + 1)
    plans: list[ComboPlan] = []
    seed_cursor = 0
    for image_index, image_path in enumerate(images):
        for prompt_index, prompt in enumerate(config.prompts):
            combo_dir = _combo_dirs(config, image_index, image_path, prompt_index, prompt)
            attempt_seeds = None
            if all_attempt_seeds is not None:
                attempt_seeds = all_attempt_seeds[seed_cursor : seed_cursor + brute_config.trials]
            seed_cursor += brute_config.trials
            rng_seed = sampling_seed_rng.randint(1, 2_147_483_647) if brute_config.randomize_attempt_seed else None
            plans.append(
                ComboPlan(
                    image_index=image_index,
                    prompt_index=prompt_index,
                    image_path=image_path,
                    prompt=prompt,
                    combo_dir=combo_dir,
                    brute_config_path=combo_dir / "combo_brute.json",
                    attempt_seeds=attempt_seeds,
                    rng_seed=rng_seed,
                )
            )

    pending: list[ComboPlan] = []
    for plan in plans:
        if config.skip_existing and (plan.combo_dir / "brute_report.json").exists():
            batch_report["results"].append(_skipped_record(plan))
        else:
            _write_combo_configs(plan.combo_dir, plan.image_path, plan.prompt, brute_data, pipeline_data)
            pending.append(plan)

    if config.parallel_combinations == 1:
        for plan in pending:
            try:
                batch_report["results"].append(_run_combo(plan))
            except Exception as exc:
                batch_report["results"].append(_failure_record(plan, exc))
            batch_report["summary"] = _summarize(batch_report["results"])
            _write_json(batch_report_path, batch_report)
    elif pending:
        with ProcessPoolExecutor(max_workers=config.parallel_combinations) as executor:
            futures = {executor.submit(_run_combo, plan): plan for plan in pending}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    batch_report["results"].append(future.result())
                except Exception as exc:
                    batch_report["results"].append(_failure_record(plan, exc))
                batch_report["summary"] = _summarize(batch_report["results"])
                _write_json(batch_report_path, batch_report)

    batch_report["results"].sort(key=lambda record: (record["image_index"], record["prompt_index"]))
    batch_report["summary"] = _summarize(batch_report["results"])
    batch_report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(batch_report_path, batch_report)
    return batch_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run brute-force search across many image/prompt combinations")
    parser.add_argument("--config", type=Path, default=Path("batch_brute.json"))
    args = parser.parse_args(argv)
    report = run_batch_brute_force(args.config)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
