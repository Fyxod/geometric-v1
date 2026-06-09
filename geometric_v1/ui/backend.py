from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .manager import PROJECT_ROOT, RunManager, safe_project_path


STATIC_DIR = Path(__file__).resolve().parent / "static"
manager = RunManager()
app = FastAPI(title="geometric-v1 UI", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    configs: dict[str, Any] | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/configs")
def read_configs() -> dict[str, Any]:
    return manager.read_base_configs()


@app.post("/api/runs/{run_type}")
def start_run(run_type: str, request: RunRequest) -> dict[str, Any]:
    if run_type not in {"perturb", "diffuse", "pipeline", "brute", "batch_brute"}:
        raise HTTPException(status_code=400, detail="invalid run type")
    try:
        return manager.start_run(run_type, request.configs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.resume_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.stop_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/runs")
def list_runs(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, Any]]:
    return manager.list_runs(limit=limit)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/report")
def get_run_report(run_id: str) -> dict[str, Any]:
    try:
        return manager.read_run_report(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/events.json")
def get_run_events(run_id: str) -> list[dict[str, Any]]:
    try:
        return manager.read_events(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(run_id: str):
    try:
        manager.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(manager.event_stream(run_id), media_type="text/event-stream")


@app.get("/api/file")
def serve_file(path: str) -> FileResponse:
    try:
        resolved = safe_project_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return FileResponse(resolved, media_type=media_type, filename=resolved.name)


@app.get("/api/project")
def project_info() -> dict[str, str]:
    return {"project_root": str(PROJECT_ROOT)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the geometric-v1 local UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("geometric_v1.ui.backend:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
