# Loss-Guided Optimization Concepts

This document explains the ideas behind `loss.json` and `geometric_v1.loss_pipeline`.

## The Problem

The old brute-force search samples perturbation parameters randomly. That is useful for exploring, but it does not learn from bad attempts. The loss-guided pipeline turns each attempt into a measured objective:

- keep the image visually close to the original before diffusion
- keep identity match high before diffusion
- make identity match low after diffusion

In other words, the attack is not trying to make the input look ruined. It is trying to create a perturbation that survives the diffusion model and causes a post-diffusion identity mismatch.

## Alpha Terms

`alpha_pre` is the identity match percentage between:

```text
original.png
perturbed.png
```

This should stay high. If `alpha_pre` is low, the perturbation has already changed or destroyed identity before diffusion. That may fool a face model, but it does not test the interesting question: can a subtle or acceptable perturbation survive Flux and alter identity afterward?

`alpha_post` is the identity match percentage between:

```text
original_diffused.png
perturbed_diffused.png
```

This should go low. It is the main attack objective. `original_diffused.png` is cached once because the original image and prompt do not change. Each iteration only needs to create and diffuse the perturbed image.

## Beta Constraints

`beta` is the set of visual similarity constraints between the original image and the perturbed image. These constraints keep the optimizer honest.

Supported beta metrics:

- `PSNR`: higher is better. It mostly measures pixel-level difference.
- `SSIM`: higher is better. It measures local structural similarity.
- `FID`: lower is better, but weak for a single image in this project.
- `LPIPS`: lower is better, optional, and only available when the `lpips` package is installed.

`loss.json` has explicit beta switches:

```json
"beta": {
  "enabled": true,
  "use_psnr": true,
  "use_ssim": true,
  "use_fid": false,
  "use_lpips": false
}
```

If a switch is `false`, that metric is not part of the loss. LPIPS is also not computed unless `use_lpips` is `true`, because it loads another neural metric model.

PSNR and SSIM are cheap and deterministic. They are useful guardrails, but they do not perfectly model human perception.

FID is normally a dataset-level metric. A proper FID compares two image distributions using deep Inception features. For a single pair of images, FID is not very meaningful. The v1 implementation uses a simple RGB Gaussian distance when enabled, so treat it as a rough extra penalty rather than a scientific score.

LPIPS is closer to perceptual similarity, but it adds another model dependency. That is why it is optional.

On the Ubuntu A6000 setup, install LPIPS only when you plan to use it:

```bash
INSTALL_LPIPS=1 bash linux-gpu/install_linux_a6000.sh
```

The default loss config keeps `use_lpips` false, so the normal working install does not change.

## The Loss

The default loss has this shape:

```text
loss =
  w_alpha_post * alpha_post
  + w_alpha_pre * max(0, alpha_pre_target - alpha_pre)^2
  + w_psnr * max(0, psnr_target - psnr)^2
  + w_ssim * max(0, ssim_target - ssim)^2
  + w_fid * max(0, fid - fid_target)^2
  + parameter regularization
```

Lower is better.

`alpha_post` is minimized directly. The other terms are penalties. For example, if `alpha_pre` is above its target, it adds no penalty. If it falls below the target, the squared penalty rises quickly.

This lets you say:

```text
Attack after diffusion, but do not destroy the source image before diffusion.
```

## Why Black-Box Optimization

The current pipeline is not differentiable end to end:

- geometric perturbations are NumPy/OpenCV/SciPy operations
- Flux diffusion is a generative model with sampling
- DeepFace models are separate verification systems
- image saving/loading happens between stages

So v1 does not pretend to do backpropagation. It treats the full pipeline as a black box:

```text
parameters -> perturb image -> diffuse image -> compare identity -> loss
```

The optimizer only sees input parameters and output loss.

## SPSA

SPSA means Simultaneous Perturbation Stochastic Approximation. It is a practical black-box optimizer.

Instead of estimating one gradient dimension at a time, SPSA samples a random direction and evaluates:

```text
loss(parameters + delta * direction)
loss(parameters - delta * direction)
```

From those two scores, it estimates a gradient for all parameters at once.

This matters because each evaluation is expensive. Every candidate can run diffusion and DeepFace. SPSA keeps each optimization step to a small number of expensive evaluations.

Key settings:

- `iterations`: number of SPSA update steps
- `learning_rate`: how far to move after estimating the gradient
- `spsa_delta`: size of the plus/minus perturbation
- `random_restarts`: how many starting points to try
- `patience`: stop after this many non-improving updates when greater than zero

## Random Search

`random_search` ignores gradients and samples candidates inside the bounds. It is simple and sometimes useful as a baseline.

If random search does as well as SPSA, the loss landscape may be noisy, the objective weights may be poorly scaled, or the parameterization may not expose the right kind of identity-changing perturbation.

## Differential Evolution

`differential_evolution` uses SciPy's population optimizer. It can handle ugly black-box objectives, but it may require many evaluations.

Because every evaluation can run Flux, it can get expensive quickly. Use it for smaller parameter spaces or shorter diffusion settings first.

## Parameter Bounds

The optimizer works in normalized `[0, 1]` space internally. `loss.json` maps that normalized vector to real parameter bounds, for example:

```json
"fft-phase": {
  "strength": [0.0, 10000.0],
  "coefficients": [4, 16]
}
```

Integer fields such as `grid` and `coefficients` are rounded before perturbations are applied.

Bounds matter a lot. If they are too narrow, the optimizer cannot find useful attacks. If they are too wide, the optimizer may find ugly image-destroying artifacts unless alpha/beta penalties are strong enough.

## Reading Results

The final `report.json` records:

- best iteration
- best parameters
- best `alpha_pre`
- best `alpha_post`
- beta metrics
- final loss
- optimizer settings
- enabled and disabled loss terms
- diffusion model used
- DeepFace models used

The `best/` folder contains the best perturbed image and best post-diffusion image. The `iterations/` folder contains every evaluated candidate with its own `metrics.json`.

Good signs:

- `alpha_pre` stays high
- PSNR and SSIM stay above thresholds
- `alpha_post` falls meaningfully below normal brute-force attempts
- the best image does not look obviously destroyed

Bad signs:

- `alpha_pre` falls with `alpha_post`
- beta penalties dominate the loss
- `alpha_post` stays high for every candidate
- only one DeepFace model changes while the others stay high

## Practical Strategy

Start small:

1. Use low diffusion steps while debugging.
2. Enable only the four supported DeepFace models.
3. Keep PSNR/SSIM constraints on.
4. Run a short SPSA job.
5. Inspect `loss_history.json`.
6. Widen bounds only for parameters that seem useful.

If Flux keeps restoring identity, the geometric parameterization may be too low-level. In that case, the next direction is to add parameters for semantic image placements, face-region masks, or accessory-like patches that Flux may preserve as content rather than remove as damage.
