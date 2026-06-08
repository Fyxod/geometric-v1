from __future__ import annotations

import time

from fastapi.testclient import TestClient

from .backend import app


def run_smoke(timeout_seconds: float = 30.0) -> dict[str, object]:
    client = TestClient(app)
    configs_response = client.get("/api/configs")
    configs_response.raise_for_status()
    configs = configs_response.json()

    start_response = client.post("/api/runs/perturb", json={"configs": {"pipeline": configs["pipeline"]}})
    start_response.raise_for_status()
    run_id = start_response.json()["run_id"]

    deadline = time.time() + timeout_seconds
    record: dict[str, object] = {}
    while time.time() < deadline:
        run_response = client.get(f"/api/runs/{run_id}")
        run_response.raise_for_status()
        record = run_response.json()
        if record["status"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.2)

    events_response = client.get(f"/api/runs/{run_id}/events.json")
    events_response.raise_for_status()
    events = events_response.json()
    return {"run_id": run_id, "status": record.get("status"), "event_count": len(events)}


def main() -> int:
    result = run_smoke()
    print(result)
    return 0 if result["status"] == "completed" and result["event_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
