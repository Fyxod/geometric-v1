from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_pil_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="RGB")
    output.save(path)


def save_pil(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
