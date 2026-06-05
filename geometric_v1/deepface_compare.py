from __future__ import annotations

import time
import bz2
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DeepFaceConfig, DEFAULT_DEEPFACE_MODELS


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
