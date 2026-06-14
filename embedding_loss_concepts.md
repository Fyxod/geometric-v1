# Embedding-Loss Concepts

This document explains the practical embedding-loss optimizer in `geometric_v1.embedding_loss_pipeline`.

The goal is deepfake prevention through geometric perturbations. The input image should still look acceptable, but the downstream edited output should become less useful for face-preserving edits. A useful outcome can be:

- identity changes after diffusion
- the requested edit fails to apply cleanly
- the generated output is degraded or meaningfully different from the clean output

The `loss3` branch continues the `loss2` embedding-loss optimizer. The stable loss2 terms still treat Flux and the configured diffusion model as a black box and compare accessible inputs, outputs, embeddings, and latents. Loss3 adds an optional, disabled-by-default Flux transformer feature term for experiments that need to look inside the active Flux denoising model.

## Identity Embeddings

Face identity models turn a face image into a vector. Similar vectors usually mean the same identity; distant vectors usually mean different identities.

The embedding-loss pipeline uses only the supported project models:

- `SFace`
- `OpenFace`
- `Facenet`
- `Facenet512`

The runner first tries direct embedding extraction with `DeepFace.represent`. If that fails for a model, it falls back to `DeepFace.verify` and records the metric source as `verify_distance`.

There are two identity checks:

- `pre_identity`: compares `original.png` and `perturbed.png`. This should stay high.
- `post_identity`: compares `original_diffused.png` and `perturbed_diffused.png`. This should become distant.

The pre term is a guardrail. The post term is an attack reward.

## VAE Latents

Diffusion models often encode images into a VAE latent space. A VAE latent is a compressed internal representation of image content.

The embedding-loss pipeline tries to reuse the active diffusion pipeline's VAE. This matters because it avoids loading a second diffusion pipeline just to compute latents.

There are two possible VAE distances:

- `input_vae`: compares original and perturbed inputs.
- `output_vae`: compares original and perturbed diffusion outputs.

`output_vae` is enabled by default because it supports the anti-edit goal. `input_vae` is disabled by default because it can fight visual stealth; if you reward input latent distance too much, the optimizer may damage the input image.

## CLIP Image Embeddings

CLIP image embeddings capture high-level visual semantics. Two images can be pixel-different but CLIP-similar if they preserve the same semantic content.

CLIP is optional and disabled by default. When enabled, the runner uses `transformers` with `openai/clip-vit-base-patch32` unless you choose another `model_id`.

CLIP can be useful when identity does not change but the generated output has drifted semantically. It can also add another model download and more runtime, so keep it off until the simpler signals are not enough.

## Flux Transformer Features

Flux does not only use the VAE image representation. During generation it runs a transformer over text and image-conditioning information while denoising toward the edited output. Those transformer hidden states are closer to the model's editing process than a final image metric or a VAE latent alone.

The optional `objective.flux_transformer` term tries to collect those hidden states during the same diffusion calls already used by the optimizer:

```text
original input -> Flux generation -> original transformer features
perturbed input -> Flux generation -> perturbed transformer features
```

It then compares pooled feature vectors. Larger feature distance becomes a reward:

```text
loss -= weight * flux_transformer_feature_distance
```

This is different from VAE latent loss:

- VAE latent loss compares compressed image representations before or after generation.
- Flux transformer feature loss compares internal activations used by the edit model while it is processing the prompt and image.

That makes the Flux feature term potentially more directly connected to edit behavior. It is also riskier. Diffusers may change private model structure, the Flux pipeline may not expose hookable blocks, and hidden tensors can be large. The implementation therefore uses guarded forward hooks, pools tensors immediately, moves pooled vectors to CPU, and records `available: false` instead of crashing when `strict` is `false`.

The `timesteps` labels are approximate. Forward hooks do not reliably expose official denoising timesteps across every Flux/Diffusers version, so the runner buckets hook calls into `early`, `middle`, and `late` by call order. Treat those buckets as a practical diagnostic, not a paper-grade timestep trace.

Changing Flux internal features does not guarantee a visible edit failure. Flux may still reconstruct identity from other signals. Use this term together with post-identity distance, output disruption, and input stealth constraints.

## Output Disruption

For deepfake prevention, identity mismatch is not the only useful success mode. If the edit becomes garbled, unstable, or fails to preserve the intended look, that is also useful.

Output disruption compares:

```text
original_diffused.png
perturbed_diffused.png
```

Supported disruption metrics include:

- pixel L2 distance
- SSIM drop, computed as `1 - SSIM`
- optional LPIPS
- output VAE distance through the VAE objective
- output CLIP distance through the CLIP objective

These terms are rewards in a lower-is-better loss, so they appear as negative loss components.

## Input Stealth

Input stealth keeps the perturbation acceptable before diffusion. Without these constraints, the optimizer can discover ugly changes that are not interesting for practical prevention.

Input stealth compares:

```text
original.png
perturbed.png
```

Default stealth terms:

- PSNR should stay above a target.
- SSIM should stay above a target.

Optional stealth terms:

- LPIPS should stay below a target.

## The Loss

The embedding loss is lower-is-better. It mixes penalties and rewards:

```text
loss =
  + input stealth penalties
  + pre-identity penalty
  - post-identity distance reward
  - output VAE distance reward
  - output CLIP distance reward
  - Flux transformer feature distance reward
  - output disruption rewards
  + parameter regularization
```

The exact terms are controlled independently in `embedding_loss.json`.

This lets you choose the strategy:

- protect the input strongly with PSNR/SSIM
- attack identity after diffusion
- reward output degradation even when identity remains similar
- add VAE or CLIP embedding pressure when useful
- add experimental Flux transformer pressure when the installed Flux pipeline exposes stable hook points

## Black-Box And Optional Flux Hooks

The optimizer still does not backpropagate through Flux. SPSA/random search are black-box optimizers over perturbation parameters. The Flux transformer feature term is observational: it reads pooled activations during generation when available, but it does not compute gradients through them.

Without Flux hooks, each candidate follows this expensive but stable loop:

```text
parameters
-> geometric perturbation
-> perturbed.png
-> configured diffusion model with same prompt
-> perturbed_diffused.png
-> identity / latent / CLIP / disruption metrics
-> scalar loss
```

SPSA uses plus and minus parameter probes to estimate a direction without gradients. This is why each SPSA iteration evaluates three candidates: plus, minus, and current.

With Flux hooks enabled, the same generation calls additionally capture pooled internal features for the original and perturbed diffusion runs. If hooks are unavailable and `strict` is `false`, the candidate still runs and `metrics.flux_transformer.available` is `false`.

## Reading Reports

Each `metrics.json` stores:

- raw metrics under `metrics`
- weighted signed terms under `loss_components`
- model/source details under identity model entries
- VAE, CLIP, and Flux backend status under `embedding_backends`
- Flux feature availability, captured layers/timestep buckets, shapes, distance, warnings, and contribution under `metrics.flux_transformer`
- optional failures under `errors`

Negative loss components are rewards. For example, a large `output_vae_distance_reward` with a negative value means the candidate increased output VAE distance, which the optimizer wants.

If one identity model fails and `strict` is `false`, that model is recorded and excluded from the average. If `strict` is `true`, that failure stops the run.
