from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import DeepFaceConfig, load_pipeline_config
from .deepface_compare import compare_images


def _write_response(handle, response: dict[str, Any]) -> None:
    handle.write(json.dumps(response) + "\n")
    handle.flush()


def _compare_request(request: dict[str, Any]) -> dict[str, Any]:
    image_a = Path(str(request["image_a"]))
    image_b = Path(str(request["image_b"]))
    config_path = request.get("config")
    deepface_config = load_pipeline_config(Path(str(config_path))).deepface if config_path else DeepFaceConfig()
    result = compare_images(image_a, image_b, deepface_config, allow_parallel=False)
    execution = result.setdefault("execution", {})
    if isinstance(execution, dict):
        execution["persistent_worker"] = True
        execution["parallel"] = False
        execution["resolved_workers"] = 1
    return result


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("type") == "shutdown":
                _write_response(protocol_out, {"id": request_id, "ok": True, "shutdown": True})
                return 0
            result = _compare_request(request)
            _write_response(protocol_out, {"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            _write_response(
                protocol_out,
                {
                    "id": request.get("id") if isinstance(locals().get("request"), dict) else None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
