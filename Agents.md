# Agents.md

## Project Context

This repository is the user's geometric anti-edit/deepfake-prevention experiment. The user wants a system that can take a normal face image and a prompt, produce a perturbed version that still looks acceptable, then test whether diffusion editing changes identity, fails, or degrades compared with the clean edit.

The user's main pain point is that Flux is surprisingly good at repairing blur, warps, and other small perturbations. A perturbation can look damaged to us, but Flux may still reconstruct the person and apply the requested edit. That means the project needs measurable, repeatable optimization rather than only manual perturbation guesses.

The end goal is to discover perturbation settings that keep the pre-diffusion image close to the original while reducing post-diffusion identity similarity or disrupting the generated edit.

## Branch Purpose: loss1

`loss1` is the first loss-guided optimization branch. It is currently aligned with `main`, but it exists as the named checkpoint for the practical black-box loss optimizer added before the later embedding-loss branches.

This branch contains:

- the full geometric-v1 baseline: perturbations, diffusion, DeepFace comparison, brute force, batch brute force, UI, and Linux GPU setup
- `loss.json`
- `run_loss_pipeline.py`
- module form: `python -m geometric_v1.loss_pipeline --config loss.json`
- sample loss config documentation under `sample_jsons/`
- `loss_concepts.md`

## Additions In This Branch

The key addition is `geometric_v1.loss_pipeline`, a black-box optimizer over geometric perturbation parameters. It does not backpropagate through Flux, DeepFace, or the perturbation stack. Instead it evaluates candidate perturbation settings and uses optimizers such as SPSA/random search/differential evolution where available.

The loss is designed around:

- `alpha_pre`: identity match between `original.png` and `perturbed.png`; this should stay high
- `alpha_post`: identity match between `original_diffused.png` and `perturbed_diffused.png`; this should go low
- beta visual constraints such as PSNR, SSIM, optional LPIPS, and an optional lightweight FID-style approximation
- parameter regularization to avoid extreme perturbation settings

The pipeline caches `original_diffused.png` once, then each iteration perturbs the input, diffuses only the perturbed image, computes metrics, writes iteration reports, and updates the optimizer state.

## Problems This Branch Is Meant To Study

- Whether geometry/frequency perturbation parameters can be optimized instead of manually guessed.
- Whether small perturbations can survive Flux edits enough to reduce post-edit identity similarity.
- How to balance input stealth against output disruption.
- Whether brute-force findings can be turned into a more directed search objective.

## Working Rules For Future Agents

- Keep `loss_pipeline.py` black-box; do not fake gradients through Flux or DeepFace.
- Keep all loss terms optional/configurable from `loss.json`.
- Preserve existing CLI commands and module forms.
- Keep report fields clear: raw metrics are not the same thing as weighted loss components.
- If adding a new metric, update `loss.json`, `sample_jsons/sample_loss.json`, README, and `loss_concepts.md`.
- Be conservative with defaults because each evaluation can trigger a costly diffusion pass.
