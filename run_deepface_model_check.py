from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geometric_v1.deepface_compare import ALL_DEEPFACE_MODELS, compare_images
from geometric_v1.samples import ensure_sample_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every DeepFace model on generated sample images")
    parser.add_argument("--output", type=Path, default=Path("output/deepface_model_check.json"))
    args = parser.parse_args()

    image_a, image_b = ensure_sample_images(Path("samples"))
    result = compare_images(image_a, image_b, models={model: True for model in ALL_DEEPFACE_MODELS})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
