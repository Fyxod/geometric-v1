from __future__ import annotations

import argparse
from pathlib import Path

from .config import DiffusionConfig
from .diffusion import edit_image
from .image_io import load_pil_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run InstructPix2Pix on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    args = parser.parse_args(argv)

    config = DiffusionConfig(
        gpu_index=args.gpu_index,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
    )
    output = edit_image(load_pil_image(args.image), args.prompt, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
