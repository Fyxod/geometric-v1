from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deepface_compare import ALL_DEEPFACE_MODELS, compare_images
from .samples import ensure_sample_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run supported DeepFace models on generated sample images")
    parser.add_argument("--output", type=Path, default=Path("output/deepface_model_check.json"))
    args = parser.parse_args(argv)

    image_a, image_b = ensure_sample_images(Path("samples"))
    result = compare_images(image_a, image_b, models={model: True for model in ALL_DEEPFACE_MODELS})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
