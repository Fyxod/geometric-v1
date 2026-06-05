from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from geometric_v1.config import load_pipeline_config
from geometric_v1.image_io import load_image, save_image
from geometric_v1.perturbations import apply_perturbation_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Create perturbed.png from pipeline.json")
    parser.add_argument("--config", type=Path, default=Path("pipeline.json"))
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    image = load_image(config.input_path)
    perturbed = apply_perturbation_pipeline(image, config.perturbations)
    output_path = config.output_dir / "perturbed.png"
    save_image(output_path, perturbed)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
