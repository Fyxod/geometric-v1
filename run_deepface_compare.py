from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geometric_v1.config import DeepFaceConfig, load_pipeline_config
from geometric_v1.deepface_compare import compare_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two images with configured DeepFace models")
    parser.add_argument("--image-a", type=Path, required=True)
    parser.add_argument("--image-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    deepface_config = load_pipeline_config(args.config).deepface if args.config else DeepFaceConfig()
    result = compare_images(args.image_a, args.image_b, deepface_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
