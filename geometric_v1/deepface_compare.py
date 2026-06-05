from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import DeepFaceConfig, DEFAULT_DEEPFACE_MODELS


ALL_DEEPFACE_MODELS = tuple(DEFAULT_DEEPFACE_MODELS.keys())
KNOWN_WEIGHT_URLS = {
    "SFace": (
        Path.home() / ".deepface" / "weights" / "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    )
}


def _match_percent(distance: float | None, threshold: float | None) -> float | None:
    if distance is None or threshold is None or threshold <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / (2.0 * threshold))))


def _ensure_known_weight(model_name: str) -> None:
    if model_name not in KNOWN_WEIGHT_URLS:
        return
    target, url = KNOWN_WEIGHT_URLS[model_name]
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)

    import requests

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def compare_images(
    image_a: Path,
    image_b: Path,
    config: DeepFaceConfig | None = None,
    models: dict[str, bool] | None = None,
) -> dict[str, Any]:
    from deepface import DeepFace

    config = config or DeepFaceConfig()
    selected_models = models or config.models
    results: dict[str, Any] = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "detector_backend": config.detector_backend,
        "distance_metric": config.distance_metric,
        "models": {},
    }

    for model_name, enabled in selected_models.items():
        if not enabled:
            results["models"][model_name] = {"enabled": False, "skipped": True}
            continue

        started = time.perf_counter()
        try:
            _ensure_known_weight(model_name)
            verification = DeepFace.verify(
                img1_path=str(image_a),
                img2_path=str(image_b),
                model_name=model_name,
                detector_backend=config.detector_backend,
                distance_metric=config.distance_metric,
                enforce_detection=config.enforce_detection,
                align=config.align,
                silent=True,
            )
            distance = float(verification.get("distance")) if verification.get("distance") is not None else None
            threshold = float(verification.get("threshold")) if verification.get("threshold") is not None else None
            results["models"][model_name] = {
                "enabled": True,
                "ok": True,
                "verified": bool(verification.get("verified")),
                "distance": distance,
                "threshold": threshold,
                "match_percent": _match_percent(distance, threshold),
                "elapsed_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            results["models"][model_name] = {
                "enabled": True,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }

    ok_values = [
        value["match_percent"]
        for value in results["models"].values()
        if value.get("ok") and value.get("match_percent") is not None
    ]
    results["summary"] = {
        "successful_models": len(ok_values),
        "mean_match_percent": sum(ok_values) / len(ok_values) if ok_values else None,
        "min_match_percent": min(ok_values) if ok_values else None,
        "max_match_percent": max(ok_values) if ok_values else None,
    }
    return results
