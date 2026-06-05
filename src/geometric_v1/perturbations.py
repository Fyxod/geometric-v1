from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import Delaunay

from .config import PerturbationStep


Array = np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _base_grid(height: int, width: int) -> tuple[Array, Array]:
    yy, xx = np.mgrid[0:height, 0:width]
    return xx.astype(np.float32), yy.astype(np.float32)


def _remap(image: Array, map_x: Array, map_y: Array, mode: str = "reflect") -> Array:
    channels = [
        map_coordinates(image[:, :, channel], [map_y, map_x], order=1, mode=mode)
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=2).astype(np.float32).clip(0.0, 1.0)


def _control_grid(height: int, width: int, grid: int) -> Array:
    if grid < 3:
        raise ValueError("grid must be at least 3")
    xs = np.linspace(0, width - 1, grid, dtype=np.float32)
    ys = np.linspace(0, height - 1, grid, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def _jitter_points(points: Array, height: int, width: int, strength: float, seed: int) -> Array:
    rng = _rng(seed)
    scale = strength * min(height, width)
    jitter = rng.normal(0.0, scale, size=points.shape).astype(np.float32)
    moved = points + jitter
    border = (
        np.isclose(points[:, 0], 0)
        | np.isclose(points[:, 0], width - 1)
        | np.isclose(points[:, 1], 0)
        | np.isclose(points[:, 1], height - 1)
    )
    moved[border] = points[border]
    moved[:, 0] = np.clip(moved[:, 0], 0, width - 1)
    moved[:, 1] = np.clip(moved[:, 1], 0, height - 1)
    return moved


def homography_warp(image: Array, strength: float, seed: int) -> Array:
    height, width = image.shape[:2]
    rng = _rng(seed)
    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    scale = strength * min(height, width)
    dst = src + rng.uniform(-scale, scale, size=(4, 2)).astype(np.float32)
    dst[:, 0] = np.clip(dst[:, 0], 0, width - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, height - 1)
    inverse = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
    xx, yy = _base_grid(height, width)
    flat = np.stack([xx.ravel(), yy.ravel(), np.ones(height * width, dtype=np.float32)])
    projected = inverse @ flat
    projected /= np.maximum(projected[2:3], 1e-8)
    return _remap(image, projected[0].reshape(height, width), projected[1].reshape(height, width))


def thin_plate_spline_warp(image: Array, strength: float, seed: int, grid: int) -> Array:
    height, width = image.shape[:2]
    src = _control_grid(height, width, grid)
    dst = _jitter_points(src, height, width, strength, seed)
    weights_x, affine_x = _tps_fit(dst, src[:, 0])
    weights_y, affine_y = _tps_fit(dst, src[:, 1])
    xx, yy = _base_grid(height, width)
    query = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    map_x = _tps_eval(query, dst, weights_x, affine_x).reshape(height, width)
    map_y = _tps_eval(query, dst, weights_y, affine_y).reshape(height, width)
    return _remap(image, map_x, map_y)


def _tps_kernel(distances: Array) -> Array:
    squared = distances * distances
    with np.errstate(divide="ignore", invalid="ignore"):
        values = squared * np.log(squared)
    values[~np.isfinite(values)] = 0.0
    return values


def _tps_fit(points: Array, values: Array) -> tuple[Array, Array]:
    count = points.shape[0]
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    kernel = _tps_kernel(distances)
    polynomial = np.column_stack([np.ones(count), points])
    system = np.zeros((count + 3, count + 3), dtype=np.float64)
    system[:count, :count] = kernel
    system[:count, count:] = polynomial
    system[count:, :count] = polynomial.T
    rhs = np.concatenate([values.astype(np.float64), np.zeros(3)])
    solved = np.linalg.solve(system + np.eye(count + 3) * 1e-6, rhs)
    return solved[:count], solved[count:]


def _tps_eval(query: Array, points: Array, weights: Array, affine: Array) -> Array:
    distances = np.linalg.norm(query[:, None, :] - points[None, :, :], axis=2)
    kernel = _tps_kernel(distances)
    polynomial = np.column_stack([np.ones(query.shape[0]), query])
    return kernel @ weights + polynomial @ affine


def delaunay_warp(image: Array, strength: float, seed: int, grid: int) -> Array:
    height, width = image.shape[:2]
    src = _control_grid(height, width, grid)
    dst = _jitter_points(src, height, width, strength, seed)
    triangulation = Delaunay(dst)
    xx, yy = _base_grid(height, width)
    query = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    simplex = triangulation.find_simplex(query)
    map_xy = query.copy()
    valid = simplex >= 0
    transform = triangulation.transform[simplex[valid]]
    delta = query[valid] - transform[:, 2]
    bary = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
    bary = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
    source_triangles = src[triangulation.simplices[simplex[valid]]]
    map_xy[valid] = np.einsum("ij,ijk->ik", bary, source_triangles)
    return _remap(image, map_xy[:, 0].reshape(height, width), map_xy[:, 1].reshape(height, width))


def fft_phase_perturb(image: Array, strength: float, seed: int, coefficients: int) -> Array:
    rng = _rng(seed)
    height, width = image.shape[:2]
    cy, cx = height // 2, width // 2
    radius_y = max(2, height // 12)
    radius_x = max(2, width // 12)
    output = np.empty_like(image)
    offsets: list[tuple[int, int]] = []
    while len(offsets) < coefficients:
        dy = int(rng.integers(-radius_y, radius_y + 1))
        dx = int(rng.integers(-radius_x, radius_x + 1))
        if dy == 0 and dx == 0:
            continue
        if (dy, dx) not in offsets and (-dy, -dx) not in offsets:
            offsets.append((dy, dx))

    for channel in range(image.shape[2]):
        spectrum = np.fft.fftshift(np.fft.fft2(image[:, :, channel]))
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        for dy, dx in offsets:
            delta = float(rng.normal(0.0, strength))
            phase[(cy + dy) % height, (cx + dx) % width] += delta
            phase[(cy - dy) % height, (cx - dx) % width] -= delta
        changed = magnitude * np.exp(1j * phase)
        output[:, :, channel] = np.real(np.fft.ifft2(np.fft.ifftshift(changed)))
    return output.astype(np.float32).clip(0.0, 1.0)


def elastic_warp(image: Array, strength: float, seed: int, sigma: float) -> Array:
    height, width = image.shape[:2]
    rng = _rng(seed)
    scale = strength * min(height, width)
    dx = gaussian_filter(rng.normal(size=(height, width)), sigma=sigma, mode="reflect")
    dy = gaussian_filter(rng.normal(size=(height, width)), sigma=sigma, mode="reflect")
    dx = dx / max(float(np.std(dx)), 1e-6) * scale * 0.35
    dy = dy / max(float(np.std(dy)), 1e-6) * scale * 0.35
    xx, yy = _base_grid(height, width)
    return _remap(image, xx + dx.astype(np.float32), yy + dy.astype(np.float32))


def rolling_shutter_warp(image: Array, step: PerturbationStep) -> Array:
    height, width = image.shape[:2]
    xx, yy = _base_grid(height, width)
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)
    sine = step.strength * np.sin(2.0 * np.pi * step.rolling_frequency * rows + step.rolling_phase)
    linear = step.rolling_shear * (rows - 0.5)
    quadratic = step.rolling_acceleration * ((rows - 0.5) ** 2 - 0.25)
    shift = width * (sine + linear + quadratic)
    return _remap(image, xx - shift[:, None].astype(np.float32), yy)


def apply_perturbation(image: Array, step: PerturbationStep) -> Array:
    if not step.enabled or step.strength <= 0:
        return image
    if step.method == "homography":
        return homography_warp(image, step.strength, step.seed)
    if step.method == "thin-plate-spline":
        return thin_plate_spline_warp(image, step.strength, step.seed, step.grid)
    if step.method == "delaunay":
        return delaunay_warp(image, step.strength, step.seed, step.grid)
    if step.method == "fft-phase":
        return fft_phase_perturb(image, step.strength, step.seed, step.coefficients)
    if step.method == "elastic":
        return elastic_warp(image, step.strength, step.seed, step.sigma)
    if step.method == "rolling-shutter":
        return rolling_shutter_warp(image, step)
    raise ValueError(f"Unknown perturbation method: {step.method}")


def apply_perturbation_pipeline(image: Array, steps: list[PerturbationStep]) -> Array:
    output = image
    for step in steps:
        output = apply_perturbation(output, step)
    return output
