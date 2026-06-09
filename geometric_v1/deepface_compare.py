from __future__ import annotations

import time
import bz2
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DeepFaceConfig, DEFAULT_DEEPFACE_MODELS
from .events import EventCallback, emit_event


ALL_DEEPFACE_MODELS = tuple(DEFAULT_DEEPFACE_MODELS.keys())


@dataclass(frozen=True)
class WeightSpec:
    target: Path
    url: str | None = None
    compression: str | None = None
    gdrive_id: str | None = None


WEIGHTS_DIR = Path.home() / ".deepface" / "weights"
KNOWN_WEIGHT_URLS = {
    "VGG-Face": WeightSpec(
        WEIGHTS_DIR / "vgg_face_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5",
    ),
    "Facenet": WeightSpec(
        WEIGHTS_DIR / "facenet_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5",
    ),
    "Facenet512": WeightSpec(
        WEIGHTS_DIR / "facenet512_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
    ),
    "OpenFace": WeightSpec(
        WEIGHTS_DIR / "openface_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5",
    ),
    "DeepFace": WeightSpec(
        WEIGHTS_DIR / "VGGFace2_DeepFace_weights_val-0.9034.h5",
        url="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
        compression="zip",
    ),
    "DeepID": WeightSpec(
        WEIGHTS_DIR / "deepid_keras_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
    ),
    "ArcFace": WeightSpec(
        WEIGHTS_DIR / "arcface_weights.h5",
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
    ),
    "Dlib": WeightSpec(
        WEIGHTS_DIR / "dlib_face_recognition_resnet_model_v1.dat",
        url="http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
        compression="bz2",
    ),
    "SFace": WeightSpec(
        WEIGHTS_DIR / "face_recognition_sface_2021dec.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    "GhostFaceNet": WeightSpec(
        WEIGHTS_DIR / "ghostfacenet_v1.h5",
        url="https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5",
    ),
    "Buffalo_L": WeightSpec(
        WEIGHTS_DIR / "buffalo_l" / "webface_r50.onnx",
        gdrive_id="1N0GL-8ehw_bz2eZQWz2b0A5XBdXdxZhg",
    ),
}


def _match_percent(distance: float | None, threshold: float | None) -> float | None:
    if distance is None or threshold is None or threshold <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / (2.0 * threshold))))


def _ensure_known_weight(model_name: str) -> None:
    if model_name not in KNOWN_WEIGHT_URLS:
        return
    spec = KNOWN_WEIGHT_URLS[model_name]
    if spec.target.exists() and spec.target.stat().st_size > 0:
        return
    spec.target.parent.mkdir(parents=True, exist_ok=True)

    if spec.gdrive_id is not None:
        import gdown

        gdown.download(id=spec.gdrive_id, output=str(spec.target), quiet=False)
        return

    if spec.url is None:
        return

    download_path = spec.target
    if spec.compression is not None:
        download_path = spec.target.with_suffix(spec.target.suffix + f".{spec.compression}")

    import requests

    with requests.get(spec.url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with download_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if spec.compression == "bz2":
        spec.target.write_bytes(bz2.decompress(download_path.read_bytes()))
    elif spec.compression == "zip":
        with zipfile.ZipFile(download_path, "r") as archive:
            archive.extractall(spec.target.parent)


def _memory_gb() -> tuple[float | None, float | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.total / (1024**3), memory.available / (1024**3)
    except Exception:
        return None, None


def _enabled_model_count(models: dict[str, bool]) -> int:
    return sum(1 for enabled in models.values() if enabled)


def resolve_deepface_workers(
    config: DeepFaceConfig,
    selected_models: dict[str, bool],
    allow_parallel: bool,
) -> tuple[int, dict[str, Any]]:
    enabled_count = _enabled_model_count(selected_models)
    cpu_count = os.cpu_count() or 1
    total_memory_gb, available_memory_gb = _memory_gb()
    metadata: dict[str, Any] = {
        "allow_parallel": allow_parallel,
        "requested_workers": config.workers,
        "enabled_models": enabled_count,
        "cpu_count": cpu_count,
        "total_memory_gb": round(total_memory_gb, 2) if total_memory_gb is not None else None,
        "available_memory_gb": round(available_memory_gb, 2) if available_memory_gb is not None else None,
    }

    if not allow_parallel or enabled_count <= 1:
        metadata["resolved_workers"] = 1
        metadata["parallel"] = False
        return 1, metadata

    if isinstance(config.workers, int):
        requested = config.workers
    else:
        workers_text = str(config.workers).strip().lower()
        requested = int(workers_text) if workers_text.isdigit() else None

    if requested is not None:
        workers = max(1, min(requested, enabled_count))
    else:
        cpu_slots = max(1, min(3, cpu_count // 4 or 1))
        memory_slots = 3
        if total_memory_gb is not None:
            memory_slots = max(1, min(3, int(total_memory_gb // 5)))
        if available_memory_gb is not None and available_memory_gb < 1.5:
            memory_slots = 1
        workers = max(1, min(3, enabled_count, cpu_slots, memory_slots))

    metadata["resolved_workers"] = workers
    metadata["parallel"] = workers > 1
    return workers, metadata


def _compare_one_model(
    image_a: Path,
    image_b: Path,
    config: DeepFaceConfig,
    model_name: str,
    event_callback: EventCallback | None = None,
    event_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    from deepface import DeepFace

    started = time.perf_counter()
    context = event_context or {}
    emit_event(event_callback, "deepface_model_running", **context, model=model_name, status="running")
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
        result = {
            "enabled": True,
            "ok": True,
            "verified": bool(verification.get("verified")),
            "distance": distance,
            "threshold": threshold,
            "match_percent": _match_percent(distance, threshold),
            "elapsed_seconds": time.perf_counter() - started,
        }
        emit_event(
            event_callback,
            "deepface_model_completed",
            **context,
            model=model_name,
            percentage=result["match_percent"],
            verified=result["verified"],
            status="completed",
            elapsed_seconds=result["elapsed_seconds"],
        )
        return model_name, result
    except Exception as exc:
        result = {
            "enabled": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }
        emit_event(
            event_callback,
            "deepface_model_error",
            **context,
            model=model_name,
            status="error",
            error=result["error"],
            elapsed_seconds=result["elapsed_seconds"],
        )
        return model_name, result


def _emit_running_mean(
    event_callback: EventCallback | None,
    event_context: dict[str, Any] | None,
    model_results: dict[str, dict[str, Any]],
) -> None:
    ok_values = [
        value["match_percent"]
        for value in model_results.values()
        if value.get("ok") and value.get("match_percent") is not None
    ]
    if not ok_values:
        return
    emit_event(
        event_callback,
        "running_mean_updated",
        **(event_context or {}),
        completed_models=len(ok_values),
        mean_match_percent=sum(ok_values) / len(ok_values),
        min_match_percent=min(ok_values),
        max_match_percent=max(ok_values),
    )


def compare_images(
    image_a: Path,
    image_b: Path,
    config: DeepFaceConfig | None = None,
    models: dict[str, bool] | None = None,
    allow_parallel: bool = False,
    event_callback: EventCallback | None = None,
    event_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or DeepFaceConfig()
    selected_models = models if models is not None else config.models
    workers, execution = resolve_deepface_workers(config, selected_models, allow_parallel)
    results: dict[str, Any] = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "detector_backend": config.detector_backend,
        "distance_metric": config.distance_metric,
        "execution": execution,
        "models": {},
    }

    enabled_models: list[str] = []
    skipped_models: dict[str, dict[str, Any]] = {}
    for model_name, enabled in selected_models.items():
        if not enabled:
            skipped_models[model_name] = {"enabled": False, "skipped": True}
            emit_event(
                event_callback,
                "deepface_model_skipped",
                **(event_context or {}),
                model=model_name,
                status="skipped",
            )
            continue
        enabled_models.append(model_name)
        emit_event(
            event_callback,
            "deepface_model_pending",
            **(event_context or {}),
            model=model_name,
            status="pending",
        )

    model_results: dict[str, dict[str, Any]] = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_compare_one_model, image_a, image_b, config, model_name, event_callback, event_context)
                for model_name in enabled_models
            ]
            for future in as_completed(futures):
                model_name, model_result = future.result()
                model_results[model_name] = model_result
                _emit_running_mean(event_callback, event_context, model_results)
    else:
        for model_name in enabled_models:
            model_name, model_result = _compare_one_model(
                image_a,
                image_b,
                config,
                model_name,
                event_callback,
                event_context,
            )
            model_results[model_name] = model_result
            _emit_running_mean(event_callback, event_context, model_results)

    for model_name in selected_models:
        if model_name in skipped_models:
            results["models"][model_name] = skipped_models[model_name]
        else:
            results["models"][model_name] = model_results[model_name]

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
