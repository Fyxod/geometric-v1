from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..batch_brute_force import run_batch_brute_force
from ..brute_force import run_brute_force
from ..operations import run_diffuse_only, run_perturb_only
from ..pipeline import run_pipeline
from .history import HistoryDB


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_OUTPUT_ROOT = PROJECT_ROOT / "output" / "ui_runs"
BASE_CONFIGS = {
    "pipeline": PROJECT_ROOT / "pipeline.json",
    "brute": PROJECT_ROOT / "brute.json",
    "batch_brute": PROJECT_ROOT / "batch_brute.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _resolve_project_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def _run_id(run_type: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{run_type}_{uuid.uuid4().hex[:8]}"


class RunContext:
    def __init__(self, run_id: str, run_type: str, run_dir: Path, db: HistoryDB) -> None:
        self.run_id = run_id
        self.run_type = run_type
        self.run_dir = run_dir
        self.db = db
        self.events_path = run_dir / "events.jsonl"
        self.stop_event = threading.Event()
        self._subscribers: list[queue.Queue[dict[str, Any] | None]] = []
        self._lock = threading.Lock()
        self._sequence = 0
        self._progress: dict[str, Any] = {}
        self._min_score: float | None = None
        self._max_score: float | None = None

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            enriched = {
                "sequence": self._sequence,
                "timestamp": _now(),
                "run_id": self.run_id,
                "ui_run_type": self.run_type,
                **event,
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(enriched) + "\n")
            self._update_progress(enriched)
            for subscriber in list(self._subscribers):
                subscriber.put(enriched)

    def _update_progress(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        self._progress["last_event"] = event_type
        self._progress["last_event_at"] = event["timestamp"]
        for key in (
            "run_number",
            "run_name",
            "status",
            "summary",
            "image_path",
            "prompt",
            "image_index",
            "prompt_index",
        ):
            if key in event:
                self._progress[key] = event[key]
        if event_type == "brute_attempt_completed" and event.get("average_match_percent") is not None:
            value = float(event["average_match_percent"])
            self._min_score = value if self._min_score is None else min(self._min_score, value)
            self._max_score = value if self._max_score is None else max(self._max_score, value)
        if event_type == "min_max_score_updated":
            min_score = event.get("min_score") or {}
            max_score = event.get("max_score") or {}
            if min_score.get("mean_percentage") is not None:
                self._min_score = float(min_score["mean_percentage"])
            if max_score.get("mean_percentage") is not None:
                self._max_score = float(max_score["mean_percentage"])
            self._progress["min_score"] = min_score
            self._progress["max_score"] = max_score
        self.db.update_run(
            self.run_id,
            progress_summary=self._progress,
            min_score=self._min_score,
            max_score=self._max_score,
        )

    def subscribe(self) -> queue.Queue[dict[str, Any] | None]:
        subscriber: queue.Queue[dict[str, Any] | None] = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any] | None]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def close_subscribers(self) -> None:
        with self._lock:
            for subscriber in list(self._subscribers):
                subscriber.put(None)
            self._subscribers.clear()


class RunManager:
    def __init__(self, output_root: Path = UI_OUTPUT_ROOT) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.db = HistoryDB(self.output_root / "ui_history.sqlite")
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._contexts: dict[str, RunContext] = {}
        self._contexts_lock = threading.Lock()

    def read_base_configs(self) -> dict[str, Any]:
        return {name: _read_json(path) for name, path in BASE_CONFIGS.items()}

    def start_run(self, run_type: str, configs: dict[str, Any] | None = None) -> dict[str, Any]:
        configs = configs or {}
        run_id = _run_id(run_type)
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        config_paths = self._write_effective_configs(run_type, run_dir, configs, resume_existing=False)
        context = RunContext(run_id, run_type, run_dir, self.db)
        self.db.create_run(
            run_id=run_id,
            run_type=run_type,
            status="queued",
            started_at=_now(),
            config_paths={key: str(value) for key, value in config_paths.items()},
            output_path=str(run_dir),
        )
        with self._contexts_lock:
            self._contexts[run_id] = context
        self._executor.submit(self._execute, context, config_paths)
        return {"run_id": run_id, "status": "queued", "output_path": str(run_dir)}

    def resume_run(self, run_id: str) -> dict[str, Any]:
        record = self.db.get_run(run_id)
        if not record:
            raise FileNotFoundError(f"unknown run id: {run_id}")
        if record["run_type"] not in {"brute", "batch_brute"}:
            raise ValueError("only brute and batch_brute runs support resume")
        with self._contexts_lock:
            existing = self._contexts.get(run_id)
            if existing and record["status"] in {"queued", "running"}:
                raise RuntimeError("run is already active")
        run_dir = Path(record["output_path"])
        config_paths = {key: Path(value) for key, value in record["config_paths"].items()}
        self._force_resume(config_paths)
        context = RunContext(run_id, record["run_type"], run_dir, self.db)
        with self._contexts_lock:
            self._contexts[run_id] = context
        self.db.update_run(run_id, status="queued", finished_at=None)
        self._executor.submit(self._execute, context, config_paths)
        return {"run_id": run_id, "status": "queued", "output_path": str(run_dir)}

    def stop_run(self, run_id: str) -> dict[str, Any]:
        context = self._contexts.get(run_id)
        if not context:
            raise FileNotFoundError(f"run is not active: {run_id}")
        context.stop_event.set()
        context.emit({"type": "stop_requested", "message": "Run will stop after the current safe boundary"})
        return {"run_id": run_id, "status": "stopping"}

    def _write_effective_configs(
        self,
        run_type: str,
        run_dir: Path,
        configs: dict[str, Any],
        resume_existing: bool,
    ) -> dict[str, Path]:
        pipeline = deepcopy(configs.get("pipeline") or _read_json(BASE_CONFIGS["pipeline"]))
        brute = deepcopy(configs.get("brute") or _read_json(BASE_CONFIGS["brute"]))
        batch = deepcopy(configs.get("batch_brute") or _read_json(BASE_CONFIGS["batch_brute"]))

        if "input" in pipeline:
            pipeline["input"] = _resolve_project_path(str(pipeline["input"]))
        pipeline["output_dir"] = str(run_dir)

        effective_pipeline = run_dir / "effective_pipeline.json"
        _write_json(effective_pipeline, pipeline)
        paths: dict[str, Path] = {"pipeline": effective_pipeline}

        if run_type in {"brute", "batch_brute"}:
            brute["pipeline_config"] = str(effective_pipeline)
            brute["output_dir"] = str(run_dir)
            if resume_existing:
                brute["resume"] = True
            effective_brute = run_dir / "effective_brute.json"
            _write_json(effective_brute, brute)
            paths["brute"] = effective_brute

        if run_type == "batch_brute":
            if "images_dir" in batch:
                batch["images_dir"] = _resolve_project_path(str(batch["images_dir"]))
            batch["pipeline_config"] = str(effective_pipeline)
            batch["brute_config"] = str(paths["brute"])
            batch["output_dir"] = str(run_dir)
            effective_batch = run_dir / "effective_batch_brute.json"
            _write_json(effective_batch, batch)
            paths["batch_brute"] = effective_batch

        return paths

    def _force_resume(self, config_paths: dict[str, Path]) -> None:
        brute_path = config_paths.get("brute")
        if brute_path and brute_path.exists():
            brute = _read_json(brute_path)
            brute["resume"] = True
            _write_json(brute_path, brute)

    def _execute(self, context: RunContext, config_paths: dict[str, Path]) -> None:
        self.db.update_run(context.run_id, status="running")
        context.emit({"type": "ui_run_started", "run_type": context.run_type})
        status = "completed"
        report_path = context.run_dir / "report.json"
        try:
            if context.run_type == "perturb":
                report = run_perturb_only(config_paths["pipeline"], event_callback=context.emit)
            elif context.run_type == "diffuse":
                report = run_diffuse_only(config_paths["pipeline"], event_callback=context.emit)
            elif context.run_type == "pipeline":
                report = run_pipeline(config_paths["pipeline"], event_callback=context.emit)
            elif context.run_type == "brute":
                report = run_brute_force(
                    config_paths["brute"],
                    event_callback=context.emit,
                    stop_requested=context.stop_event.is_set,
                )
                report_path = context.run_dir / "brute_report.json"
            elif context.run_type == "batch_brute":
                report = run_batch_brute_force(
                    config_paths["batch_brute"],
                    event_callback=context.emit,
                    stop_requested=context.stop_event.is_set,
                )
                report_path = context.run_dir / "batch_report.json"
            else:
                raise ValueError(f"unknown run type: {context.run_type}")

            if report_path.exists() and report_path.name != "report.json":
                shutil.copyfile(report_path, context.run_dir / "report.json")
            elif not report_path.exists():
                _write_json(context.run_dir / "report.json", report)
                report_path = context.run_dir / "report.json"
            if context.stop_event.is_set() or report.get("status") == "stopped":
                status = "stopped"
            context.emit({"type": "ui_run_completed", "run_type": context.run_type, "status": status})
        except Exception as exc:
            status = "failed"
            report_path = context.run_dir / "report.json"
            _write_json(report_path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            context.emit({"type": "ui_run_failed", "run_type": context.run_type, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.db.update_run(
                context.run_id,
                status=status,
                finished_at=_now(),
                report_path=str(report_path),
            )
            context.close_subscribers()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.list_runs(limit=limit)

    def get_run(self, run_id: str) -> dict[str, Any]:
        record = self.db.get_run(run_id)
        if not record:
            raise FileNotFoundError(f"unknown run id: {run_id}")
        return record

    def read_run_report(self, run_id: str) -> dict[str, Any]:
        record = self.get_run(run_id)
        report_path = record.get("report_path")
        if not report_path:
            candidate = Path(record["output_path"]) / "report.json"
            report_path = str(candidate)
        path = Path(report_path)
        if not path.exists():
            return {"status": record["status"], "message": "report is not available yet"}
        return _read_json(path)

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get_run(run_id)
        path = Path(record["output_path"]) / "events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    async def event_stream(self, run_id: str):
        record = self.get_run(run_id)
        for event in self.read_events(run_id):
            yield f"data: {json.dumps(event)}\n\n"
        if record["status"] not in {"queued", "running"}:
            return
        context = self._contexts.get(run_id)
        if context is None:
            return
        subscriber = context.subscribe()
        try:
            while True:
                event = await asyncio.to_thread(subscriber.get)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            context.unsubscribe(subscriber)


def safe_project_path(value: str) -> Path:
    path = Path(value).resolve()
    root = PROJECT_ROOT.resolve()
    if path == root or root in path.parents:
        return path
    raise ValueError("path is outside the geometric-v1 project")
