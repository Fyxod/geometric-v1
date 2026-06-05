from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_pipeline_config
from .image_io import load_image, save_image
from .perturbations import apply_perturbation_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create perturbed.png from pipeline.json")
    parser.add_argument("--config", type=Path, default=Path("pipeline.json"))
    args = parser.parse_args(argv)

    config = load_pipeline_config(args.config)
    image = load_image(config.input_path)
    perturbed = apply_perturbation_pipeline(image, config.perturbations)
    output_path = config.output_dir / "perturbed.png"
    save_image(output_path, perturbed)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
