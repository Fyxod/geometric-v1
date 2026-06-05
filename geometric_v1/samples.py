from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def _draw_sample_face(path: Path, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (256, 256), (220, 224, 232))
    draw = ImageDraw.Draw(image)
    draw.ellipse((58 + offset, 34, 198 + offset, 190), fill=(196, 144, 108), outline=(80, 60, 50), width=3)
    draw.ellipse((92 + offset, 92, 108 + offset, 108), fill=(25, 25, 30))
    draw.ellipse((150 + offset, 92, 166 + offset, 108), fill=(25, 25, 30))
    draw.arc((102 + offset, 118, 158 + offset, 164), 20, 160, fill=(110, 50, 55), width=4)
    draw.polygon([(128 + offset, 108), (118 + offset, 132), (138 + offset, 132)], fill=(164, 105, 85))
    draw.arc((60 + offset, 30, 198 + offset, 110), 180, 360, fill=(40, 32, 28), width=16)
    image.save(path)


def ensure_sample_images(sample_dir: Path) -> tuple[Path, Path]:
    image_a = sample_dir / "sample_face_a.png"
    image_b = sample_dir / "sample_face_b.png"
    if not image_a.exists():
        _draw_sample_face(image_a, offset=0)
    if not image_b.exists():
        _draw_sample_face(image_b, offset=4)
    return image_a, image_b
