# Agents.md

## Project Context

This repository is the user's geometric anti-edit/deepfake-prevention experiment. The user is trying to answer a practical question: can a face image be subtly perturbed so that it still looks normal and matches the same person before editing, but causes an image-to-image diffusion edit to fail, degrade, or change identity afterward?

The recurring problem is that Flux can repair or ignore many perturbations. A perturbed image may be blurry, warped, or frequency-shifted, yet the diffusion model can still reconstruct the person and apply prompts like "add sunglasses." This branch moves beyond plain brute force and scalar identity loss by comparing more embedding and output signals around the diffusion process.

The end goal is a configurable research pipeline that can search for anti-edit perturbations, explain why a candidate was good or bad, and produce reproducible reports for every attempt.

## Branch Purpose: loss2

`loss2` builds the practical embedding-loss anti-edit pipeline. It continues the `loss1` black-box optimization direction, but adds richer objective terms around identity embeddings, active diffusion VAE latents, optional CLIP image embeddings, input stealth, and output disruption.

This branch contains everything from the main/loss1 baseline plus:

- `embedding_loss.json`
- `run_embedding_loss_pipeline.py`
- module form: `python -m geometric_v1.embedding_loss_pipeline --config embedding_loss.json`
- `geometric_v1/embedding_loss_pipeline.py`
- `embedding_loss_concepts.md`
- `sample_jsons/sample_embedding_loss.json`

## Additions In This Branch

The key addition is a practical embedding-loss optimizer. It remains black-box with respect to Flux generation and DeepFace, but it uses more signals than `loss.json`.

Objective areas include:

- input stealth: PSNR, SSIM, optional LPIPS between `original.png` and `perturbed.png`
- identity: pre-diffusion identity should stay high; post-diffusion identity distance is rewarded
- VAE latents: optional input VAE distance and default output VAE distance from the active diffusion pipeline when available
- CLIP image embeddings: optional semantic image distance through `transformers`
- output disruption: pixel L2, SSIM drop, and optional LPIPS between clean and perturbed diffusion outputs
- parameter regularization

The runner caches the clean original diffusion output and original-side embeddings/latents where possible, then evaluates perturbed candidates through the same prompt.

## Problems This Branch Is Meant To Study

- Whether output-level disruption can succeed even when identity metrics alone do not move enough.
- Whether VAE or CLIP embedding distances expose useful pressure that DeepFace misses.
- Whether the optimizer can find subtle perturbations that Flux does not simply repair.
- Whether a candidate can remain stealthy before diffusion while becoming disruptive after diffusion.

## Known Limitations

- This branch does not hook Flux transformer hidden states.
- It does not backpropagate through Flux, VAE, CLIP, or DeepFace.
- CLIP and LPIPS are optional because they add downloads/runtime.
- DeepFace model failures should be recorded and skipped unless strict mode is enabled.

## Working Rules For Future Agents

- Do not rewrite the working `loss_pipeline.py` when changing embedding loss.
- Keep `embedding_loss_pipeline.py` black-box unless working on a later branch that explicitly adds internal Flux hooks.
- Keep the supported DeepFace model set to `SFace`, `OpenFace`, `Facenet`, and `Facenet512`.
- Keep `embedding_loss.json`, sample JSONs, README, and `embedding_loss_concepts.md` synchronized.
- Reports should clearly separate raw metrics from signed/weighted `loss_components`.
- Any new expensive metric should be disabled by default.
