# Agents.md

## Project Context

This repository is the user's geometric anti-edit/deepfake-prevention experiment. The user is trying to make an input face image remain visually acceptable and recognizable before editing, while causing downstream diffusion edits to fail, degrade, or change identity after the edit.

The core problem is that modern image-to-image diffusion models, especially Flux, are very good at repairing small geometric or frequency perturbations. Simple RGB/circular pixel attacks were not enough. The project moved toward structured geometric perturbations, brute-force search, batch search, and loss-guided optimization so the user can test which perturbations survive diffusion and disrupt identity.

The end goal is a practical experiment platform:

- take an image and prompt
- apply configurable geometric/frequency perturbations
- run the same prompt through the selected diffusion model
- compare clean-vs-perturbed outputs with DeepFace and visual metrics
- search or optimize perturbation settings automatically
- keep all runs reproducible through JSON configs and reports

## Branch Purpose: main

`main` is the stable integration branch for the current project. It should remain the safest branch for normal CLI and UI use.

This branch contains:

- root package layout under `geometric_v1/` with no `src/` layout
- geometric perturbations: homography, thin-plate spline, Delaunay, FFT phase, elastic warp, and rolling shutter
- `pipeline.json`, `brute.json`, and `batch_brute.json` as source-of-truth configs
- full pipeline, perturb-only, diffusion-only, brute-force, and batch brute-force CLIs
- Flux.2 Klein and InstructPix2Pix diffusion selection
- DeepFace support limited to `SFace`, `OpenFace`, `Facenet`, and `Facenet512`
- persistent DeepFace worker support for faster brute-force loops
- safe resume/continuation behavior for brute-force and batch brute-force runs
- local FastAPI dashboard on top of the existing runners
- Ubuntu A6000/no-root install instructions and tuned Linux GPU sample configs
- `loss.json` and `geometric_v1.loss_pipeline`, the first black-box loss-guided optimizer

## Current Additions In This Branch

The latest main-line addition is the `loss1` black-box optimizer work. It optimizes geometric perturbation parameters with SPSA/random search while balancing:

- high pre-diffusion identity match
- low post-diffusion identity match
- PSNR/SSIM/optional LPIPS/FID-style visual constraints
- parameter regularization

It caches `original_diffused.png` once and evaluates perturbed candidates against it.

## Working Rules For Future Agents

- Do not break these CLI entry points:
  - `python run_pipeline.py --config pipeline.json`
  - `python run_brute_force.py --config brute.json`
  - `python run_batch_brute_force.py --config batch_brute.json`
  - `python run_loss_pipeline.py --config loss.json`
  - module forms under `python -m geometric_v1.*`
- Keep JSON configs, sample JSONs, README sections, and reports in sync.
- Keep temporary UI overrides separate from permanent JSON files.
- Preserve existing output folder structures unless the user explicitly asks for a migration.
- Treat Flux as expensive and mostly black-box on this branch.
- If a DeepFace model errors, record the error and avoid counting it in averages unless strict behavior is explicitly requested.
- Prefer small, testable changes because the user runs long GPU experiments on remote machines.
