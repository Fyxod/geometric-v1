from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image

from .config import DiffusionConfig


def resolve_device(config: DiffusionConfig) -> str:
    if config.cpu:
        return "cpu"
    if config.device != "auto":
        return config.device
    if torch.cuda.is_available():
        return f"cuda:{config.gpu_index}"
    return "cpu"


def _resize_for_diffusion(image: Image.Image, max_size: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(max_size / max(width, height), 1.0)
    width = max(64, int(width * scale) // 8 * 8)
    height = max(64, int(height * scale) // 8 * 8)
    return image.resize((width, height), Image.Resampling.LANCZOS)


@lru_cache(maxsize=2)
def _load_pipe(model_id: str, device: str):
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def edit_image(image: Image.Image, prompt: str, config: DiffusionConfig) -> Image.Image:
    device = resolve_device(config)
    pipe = _load_pipe(config.model_id, device)
    prepared = _resize_for_diffusion(image, config.max_size)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    result = pipe(
        prompt=prompt,
        image=prepared,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale,
        generator=generator,
    )
    return result.images[0].convert("RGB")
