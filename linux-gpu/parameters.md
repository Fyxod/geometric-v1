# A6000 Parameter Profile

This file lists the intentional changes in `linux-gpu/*.json` compared with the default project configs.

## `pipeline.json`

- `input`: changed to `../samples/image.png` so it resolves correctly from `linux-gpu/pipeline.json`.
- `output_dir`: changed to `../output/linux_gpu_pipeline`.
- `seed`: changed to `42` for a clean reproducible Linux GPU profile.
- `homography.enabled`: changed to `true`.
- `homography.strength`: set to `0.035`.
- `thin-plate-spline.strength`: set to `0.004`.
- `thin-plate-spline.grid`: set to `9`.
- `delaunay.strength`: set to `0.009`.
- `delaunay.grid`: set to `10`.
- `fft-phase.strength`: set to `0.22`.
- `fft-phase.coefficients`: set to `12`.
- `elastic.strength`: set to `0.014`.
- `elastic.sigma`: set to `16.0`.
- `rolling-shutter.strength`: set to `0.01`.
- `rolling-shutter.rolling_frequency`: set to `1.8`.
- `rolling-shutter.rolling_shear`: set to `0.035`.
- `rolling-shutter.rolling_acceleration`: set to `0.005`.
- `diffusion.cpu`: set to `false`.
- `diffusion.device`: kept as `auto` so PyTorch chooses CUDA when available.
- `diffusion.gpu_index`: set to `0`, which is typical on a single-GPU server.
- `diffusion.num_inference_steps`: increased to `20`.
- `diffusion.max_size`: increased to `768`.
- `diffusion.seed`: set to `42`.
- `deepface.workers`: set to `3`.
- All DeepFace model booleans remain `true`.

Rationale: `max_size=768` and `20` diffusion steps are much more comfortable on a 48 GB A6000 than on the laptop profile, while still leaving room for DeepFace and framework overhead. I did not set `max_size=1024` by default because InstructPix2Pix memory use can jump sharply with resolution, especially when running brute force repeatedly.

## `brute.json`

- `pipeline_config`: points to `pipeline.json` inside this folder.
- `output_dir`: changed to `../output/linux_gpu_brute_force`.
- `trials`: set to `500`.
- `resume`: kept as `true`.
- `homography.strength`: `[0.005, 0.06]`.
- `thin-plate-spline.strength`: `[0.001, 0.006]`.
- `thin-plate-spline.grid`: `[5, 11]`.
- `delaunay.strength`: `[0.002, 0.012]`.
- `delaunay.grid`: `[6, 14]`.
- `fft-phase.strength`: `[0.05, 0.35]`.
- `fft-phase.coefficients`: `[4, 18]`.
- `elastic.strength`: `[0.002, 0.02]`.
- `elastic.sigma`: `[8.0, 24.0]`.
- `rolling-shutter.strength`: `[0.0, 0.014]`.
- `rolling-shutter.rolling_frequency`: `[0.5, 3.5]`.
- `rolling-shutter.rolling_shear`: `[0.0, 0.08]`.
- `rolling-shutter.rolling_acceleration`: `[-0.04, 0.04]`.

Rationale: the ranges are wider than the small local/laptop profile but avoid the extreme `fft-phase` values in the current root config. This should produce more stable image outputs and fewer unusable samples during long brute-force runs.

## `batch_brute.json`

- `brute_config`: points to `brute.json` inside this folder.
- `pipeline_config`: points to `pipeline.json` inside this folder.
- `images_dir`: set to `../samples/batch`.
- `prompts`: expanded to three prompts.
- `output_dir`: changed to `../output/linux_gpu_batch_brute`.
- `parallel_combinations`: kept at `1`.

Rationale: keeping `parallel_combinations=1` prevents several diffusion pipelines from competing for the same A6000 memory. The project already batches the two images within each pipeline run on GPU, so this is the safer default.

## When To Increase Parameters

After a successful short run, these are reasonable next steps:

- Try `diffusion.max_size=896`.
- Try `diffusion.num_inference_steps=30`.
- Try `brute.trials=1000` or more.
- Try `batch_brute.parallel_combinations=2` only if `nvidia-smi` shows plenty of free VRAM during single-combo runs.

Do not increase all of these at once. Change one parameter, run a short test, and check `nvidia-smi`.
