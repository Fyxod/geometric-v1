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
def _load_instruct_pix2pix_pipe(model_id: str, device: str):
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


def _torch_dtype(name: str, device: str) -> torch.dtype:
    if not device.startswith("cuda"):
        return torch.float32
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype for FLUX.2 Klein: {name}")


@lru_cache(maxsize=2)
def _load_flux2_klein_pipe(model_id: str, device: str, dtype_name: str, cpu_offload: bool):
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline is not available in this diffusers install. "
            "Install the current Diffusers main branch, for example: "
            "python -m pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    dtype = _torch_dtype(dtype_name, device)
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def selected_diffusion_model(config: DiffusionConfig) -> dict[str, str]:
    return {
        "name": config.selected_model,
        "model_id": config.selected_model_id,
    }


def _flux_dimensions(image: Image.Image, config: DiffusionConfig) -> tuple[int, int]:
    flux = config.flux2_klein
    if flux.height is not None and flux.width is not None:
        height = max(64, flux.height // 8 * 8)
        width = max(64, flux.width // 8 * 8)
        return height, width
    prepared = _resize_for_diffusion(image, flux.max_size)
    width, height = prepared.size
    return height, width


def _edit_instruct_pix2pix_image(image: Image.Image, prompt: str, config: DiffusionConfig) -> Image.Image:
    device = resolve_device(config)
    pipe = _load_instruct_pix2pix_pipe(config.model_id, device)
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


def _edit_flux2_klein_image(image: Image.Image, prompt: str, config: DiffusionConfig) -> Image.Image:
    device = resolve_device(config)
    flux = config.flux2_klein
    pipe = _load_flux2_klein_pipe(flux.model_id, device, flux.torch_dtype, flux.cpu_offload)
    height, width = _flux_dimensions(image, config)
    prepared = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    generator = torch.Generator(device=device).manual_seed(flux.seed)
    kwargs = {
        "image": prepared,
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_inference_steps": flux.num_inference_steps,
        "guidance_scale": flux.guidance_scale,
        "generator": generator,
        "max_sequence_length": flux.max_sequence_length,
        "text_encoder_out_layers": flux.text_encoder_out_layers,
    }
    if flux.sigmas is not None:
        kwargs["sigmas"] = flux.sigmas
    result = pipe(**kwargs)
    return result.images[0].convert("RGB")


def edit_image(image: Image.Image, prompt: str, config: DiffusionConfig) -> Image.Image:
    if config.selected_model == "flux2_klein":
        return _edit_flux2_klein_image(image, prompt, config)
    return _edit_instruct_pix2pix_image(image, prompt, config)


def _edit_instruct_pix2pix_images(images: list[Image.Image], prompt: str, config: DiffusionConfig) -> list[Image.Image]:
    if not images:
        return []
    device = resolve_device(config)
    pipe = _load_instruct_pix2pix_pipe(config.model_id, device)
    prepared = [_resize_for_diffusion(image, config.max_size) for image in images]
    generators = [
        torch.Generator(device=device).manual_seed(config.seed + index)
        for index in range(len(prepared))
    ]
    result = pipe(
        prompt=[prompt] * len(prepared),
        image=prepared,
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale,
        generator=generators,
    )
    return [image.convert("RGB") for image in result.images]


def _edit_flux2_klein_images(images: list[Image.Image], prompt: str, config: DiffusionConfig) -> list[Image.Image]:
    if not images:
        return []
    device = resolve_device(config)
    flux = config.flux2_klein
    pipe = _load_flux2_klein_pipe(flux.model_id, device, flux.torch_dtype, flux.cpu_offload)
    height, width = _flux_dimensions(images[0], config)
    prepared = [image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS) for image in images]
    generators = [
        torch.Generator(device=device).manual_seed(flux.seed + index)
        for index in range(len(prepared))
    ]
    kwargs = {
        "image": prepared,
        "prompt": [prompt] * len(prepared),
        "height": height,
        "width": width,
        "num_inference_steps": flux.num_inference_steps,
        "guidance_scale": flux.guidance_scale,
        "generator": generators,
        "max_sequence_length": flux.max_sequence_length,
        "text_encoder_out_layers": flux.text_encoder_out_layers,
    }
    if flux.sigmas is not None:
        kwargs["sigmas"] = flux.sigmas
    result = pipe(**kwargs)
    return [image.convert("RGB") for image in result.images]


def edit_images(images: list[Image.Image], prompt: str, config: DiffusionConfig) -> list[Image.Image]:
    if config.selected_model == "flux2_klein":
        return _edit_flux2_klein_images(images, prompt, config)
    return _edit_instruct_pix2pix_images(images, prompt, config)
