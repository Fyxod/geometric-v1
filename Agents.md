# Agents.md

## Project Context

This repository is the user's geometric anti-edit/deepfake-prevention experiment. The user is testing whether carefully chosen geometric and frequency perturbations can make a face image resistant to identity-preserving diffusion edits.

The central problem is that Flux is extremely robust. It can often repair blurry or warped inputs and still apply the prompt while preserving the person's identity. The user wants more than brute force: they want optimization signals that are closer to how Flux actually processes the image, while still keeping the input visually acceptable.

The end goal is a reproducible platform for anti-edit experiments:

- preserve identity and visual quality before diffusion
- disrupt identity, structure, or edit quality after diffusion
- compare runs with DeepFace and visual metrics
- search automatically across geometric perturbation parameters
- record exactly which settings, seeds, models, and metrics produced each result

## Branch Purpose: loss3

`loss3` is a continuation of `loss2`. It keeps the working embedding-loss pipeline intact and adds an optional, disabled-by-default Flux transformer/internal feature loss.

This branch contains everything from `loss2` plus:

- `geometric_v1/flux_features.py`
- `objective.flux_transformer` blocks in `embedding_loss.json` and `sample_jsons/sample_embedding_loss.json`
- Flux transformer feature reporting in `geometric_v1.embedding_loss_pipeline`
- README documentation that explicitly says loss3 continues loss2
- expanded `embedding_loss_concepts.md` coverage for Flux internal features

## Additions In This Branch

The new Flux transformer term tries to compare internal Flux hidden representations between the clean original diffusion run and each perturbed diffusion run.

When enabled, the implementation:

- uses guarded PyTorch forward hooks on discovered Flux transformer blocks
- captures features during the normal original and perturbed diffusion calls
- pools hidden tensors immediately
- moves pooled vectors to CPU
- compares original and candidate pooled features with cosine or normalized L2 distance
- rewards larger internal feature distance with `loss -= weight * distance`
- reports availability, selected layers, approximate timestep buckets, shapes, warnings, errors, distance, and loss contribution

It is disabled by default because Flux/Diffusers internals can change and because hidden activations can increase GPU memory use.

## Problems This Branch Is Meant To Study

- Whether Flux internal feature distance correlates better with edit failure than output-only metrics.
- Whether perturbations can disrupt Flux's denoising/edit path even when final image metrics look subtle.
- Whether hidden-feature pressure helps when DeepFace, VAE, CLIP, or pixel disruption signals are too weak.
- Whether the model is repairing perturbations because its internal representations remain too similar.

## Known Limitations

- This is still not end-to-end backpropagation.
- The hooks are observational; SPSA/random search still choose parameter updates.
- Only denoising-time forward-hook capture is implemented.
- Stable pre-generation input feature capture is reported as unavailable unless a future Diffusers API exposes it.
- Timestep labels are approximate buckets based on hook call order.
- If hooks fail and `strict` is `false`, the metric reports `available: false` and the run continues.
- If hooks fail and `strict` is `true`, the run should stop with a clear error.

## Working Rules For Future Agents

- Do not break the practical `embedding_loss_pipeline` behavior from `loss2`.
- Keep Flux transformer hooks optional and disabled by default.
- Avoid brittle assumptions about private Diffusers attributes unless heavily guarded.
- Do not store large GPU tensors in reports or long-lived state.
- Preserve existing CLI entry points and output layouts.
- Update `embedding_loss.json`, sample JSONs, README, and concepts docs for any objective-shape change.
- Treat this branch as experimental: clean failure handling matters more than forcing hooks to work on every installed Flux version.
