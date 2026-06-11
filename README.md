# Geometric V1

`geometric-v1` is a full image-to-report pipeline:

1. Read one input image and a text prompt from `pipeline.json`.
2. Build `perturbed.png` by applying enabled geometric/frequency perturbations in order.
3. Send both `original.png` and `perturbed.png` through the enabled diffusion model with the same prompt.
4. Compare `original_diffused.png` and `perturbed_diffused.png` with enabled DeepFace models.
5. Write all four images plus `report.json` to the configured output folder.

## Setup

Python 3.11 is recommended for this project because it is a stable intersection for PyTorch, diffusers, TensorFlow, and DeepFace on Windows.

For Ubuntu A6000 or similar 48 GB/50 GB-class GPU servers, use the dedicated setup bundle:

```bash
bash linux-gpu/install_linux_a6000.sh
source .venv-linux-gpu/bin/activate
python -m geometric_v1.pipeline --config linux-gpu/pipeline.json
```

The Ubuntu profile lives in `linux-gpu/` and includes:

- `Readme.md`: Ubuntu A6000 installation and run instructions
- `install_linux_a6000.sh`: no-root installer for Python 3.11 env, CUDA PyTorch, and project dependencies
- `constraints-a6000.txt`: Linux GPU pip constraints that prevent resolver backtracking on the A6000 dependency stack
- `pipeline.json`, `brute.json`, `batch_brute.json`: A6000-oriented configs
- `parameters.md`: explanation of every parameter change

Before installing Python packages on a new Windows laptop, install:

- Python 3.11.x
- Git
- NVIDIA driver and CUDA-compatible PyTorch only if you want GPU diffusion

```powershell
cd path\to\geometric-v1
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the correct PyTorch wheel for your CPU/GPU from https://pytorch.org/get-started/locally/ here.
python -m pip install -r requirements.txt
python -m pip install "typing-extensions>=4.14,<5"
python -m pip install -r requirements-ui.txt
```

This project pins `numpy>=1.22,<=1.24.3` because TensorFlow 2.12.1 declares that exact compatible range. TensorFlow 2.12.1 is kept for DeepFace compatibility. PyTorch needs a newer `typing-extensions`, so the final `typing-extensions` command is intentional even though TensorFlow's package metadata asks for an older version.

`albumentations` is pinned to `1.3.1` because `albumentations` 2.x requires `numpy>=1.24.4`, which conflicts with TensorFlow 2.12.1's `numpy<=1.24.3` requirement.

`requirements.txt` installs Diffusers from the current Hugging Face GitHub branch because `Flux2KleinPipeline` is not available in older stable Diffusers releases. That means Git must be available before running `python -m pip install -r requirements.txt`.

The local dashboard adds `fastapi` and `uvicorn[standard]`. They live in `requirements-ui.txt` so pip does not try to solve TensorFlow's older `typing-extensions` metadata and FastAPI's newer `typing-extensions` metadata in the same transaction. Install core requirements first, then the final `typing-extensions` override, then UI requirements.

PyTorch is not listed directly in `requirements.txt` because the right wheel depends on the laptop's CPU/GPU/CUDA setup. Install the correct PyTorch wheel before `requirements.txt`; otherwise packages such as `accelerate` may cause pip to pull or replace Torch with a default build. The `linux-gpu/install_linux_a6000.sh` script protects the installed `torch`, `torchvision`, and `torchaudio` versions with a temporary constraints file before installing the rest of the requirements. On Ubuntu A6000 installs it also passes `linux-gpu/constraints-a6000.txt` to prevent pip `resolution-too-deep` backtracking across the Diffusers, Transformers, TensorFlow, and DeepFace stack. If you change PyTorch after installing requirements, rerun:

```powershell
python -m pip install "typing-extensions>=4.14,<5"
python -m pip install -r requirements-ui.txt
```

For a CPU-only laptop, choose the CPU PyTorch command from the official selector, set `"cpu": true` in `pipeline.json`, and expect diffusion to be slow. For an NVIDIA GPU laptop, choose the CUDA PyTorch command that matches your driver, keep `"cpu": false`, and check `diffusion.resolved_device` in `report.json`.

Verify the install:

```powershell
python -c "import numpy, tensorflow as tf, torch; print('numpy', numpy.__version__); print('tensorflow', tf.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected NumPy on this stack is `1.24.3` or lower within the pinned range.

## Project Layout

The package is directly at the repository root:

```text
geometric_v1/
run_pipeline.py
run_perturbations.py
run_diffusion.py
run_deepface_compare.py
run_deepface_model_check.py
run_brute_force.py
run_batch_brute_force.py
run_loss_pipeline.py
run_ui.py
pipeline.json
brute.json
batch_brute.json
loss.json
sample_jsons/
```

There is no `src/` layout.

## Local UI

The project includes a local FastAPI dashboard on top of the existing Python runners. It does not replace the CLI commands and it does not duplicate pipeline, brute-force, or batch brute-force logic.

Start it from the repository root:

```powershell
python run_ui.py --host 127.0.0.1 --port 7860
```

Module form:

```powershell
python -m geometric_v1.ui.backend --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

The UI has tabs for:

- Perturb
- Diffuse
- Pipeline
- Brute Force
- Batch Brute Force
- History

Architecture:

- `geometric_v1/ui/backend.py` exposes the FastAPI API and serves the static UI.
- `geometric_v1/ui/manager.py` starts runs in background threads, writes effective configs, streams events, and indexes history.
- `geometric_v1/ui/history.py` stores lightweight run metadata in SQLite.
- `geometric_v1/ui/static/` contains the browser dashboard. There is no Node or frontend build step.

The UI reads `pipeline.json`, `brute.json`, and `batch_brute.json` as the source of truth. The JSON editors in the browser let you make temporary changes before starting a run. Those changes are saved only as effective configs for that UI run and are not written back to the permanent JSON files. Permanent changes should still be made manually in the repo JSON files.

Each UI-triggered run writes to:

```text
output/ui_runs/<run_id>/
  effective_pipeline.json
  effective_brute.json
  effective_batch_brute.json
  events.jsonl
  report.json
```

Only the effective configs that apply to the run type are written. For example, a perturb-only run writes `effective_pipeline.json`, while a batch brute-force run writes all three.

History storage:

- SQLite index: `output/ui_runs/ui_history.sqlite`
- Actual images, reports, sampled configs, brute outputs, and batch outputs remain in normal output folders under each UI run directory.
- The History tab can inspect previous runs after restarting the UI because it reads the SQLite index plus saved reports and `events.jsonl`.

Event streaming:

- The backend streams structured events through Server-Sent Events at `/api/runs/<run_id>/events`.
- The same events are appended to `events.jsonl`.
- CLI runs ignore events because no callback is supplied.

Events include run lifecycle, perturbation, diffusion, image writes, brute attempt state, DeepFace model state, running mean updates, min/max score updates, batch combo state, and log/status messages.

Brute-force UI behavior:

- Shows the current run number and whether the attempt is new, skipped, resumed, successful, unsuccessful, or failure.
- Shows sampled perturbation options from the current `sampled_config.json` event.
- Shows original, perturbed, original diffused, and perturbed diffused image boxes. Empty boxes are shown before files exist, and image boxes update when `image_written` events arrive.
- Shows every enabled DeepFace model as pending, running, completed, or error.
- Updates the mean as DeepFace model results arrive.
- Tracks successful, unsuccessful, failures, skipped, resumed, lowest mean, and highest mean.

Batch brute-force UI behavior:

- Shows current image/prompt combination metadata.
- Streams the same brute-force attempt and DeepFace events for the active combination.
- Tracks queued, running, completed, skipped, resumed, and failed combo events.
- Supports inspection of completed runs through the History tab and saved reports.

Resume and stopping:

- The Stop button requests a safe stop. Brute force stops after the current attempt boundary; batch brute force stops at the next safe combo/attempt boundary.
- Stopping does not delete completed run folders.
- Resume is available from History for brute and batch brute runs. It reruns the saved effective config with brute-force resume enabled, so completed folders are skipped and incomplete folders are archived according to the normal resume rules.

Backend endpoints:

- `GET /api/configs`
- `POST /api/runs/perturb`
- `POST /api/runs/diffuse`
- `POST /api/runs/pipeline`
- `POST /api/runs/brute`
- `POST /api/runs/batch_brute`
- `POST /api/runs/<run_id>/stop`
- `POST /api/runs/<run_id>/resume`
- `GET /api/runs`
- `GET /api/runs/<run_id>`
- `GET /api/runs/<run_id>/report`
- `GET /api/runs/<run_id>/events`
- `GET /api/runs/<run_id>/events.json`
- `GET /api/file?path=<absolute-project-path>`

Run a lightweight backend smoke test:

```powershell
python -m geometric_v1.ui.smoke
```

## Full Pipeline

Edit `pipeline.json`, especially:

- `input`
- `output_dir`
- `prompt`
- perturbation strengths
- `diffusion.cpu` if you want to force a CPU run
- `deepface.workers` if you want to override GPU-mode DeepFace worker slots
- DeepFace model booleans

Then run:

```powershell
python run_pipeline.py --config pipeline.json
```

Equivalent module form:

```powershell
python -m geometric_v1.pipeline --config pipeline.json
```

Output folder contents:

```text
original.png
perturbed.png
original_diffused.png
perturbed_diffused.png
report.json
```

## Brute Force Search

`brute.json` runs random perturbation trials while keeping `pipeline.json` as the source of truth for the input image, prompt, diffusion settings, CPU/GPU flag, DeepFace models, and which perturbation methods are enabled.

Run:

```powershell
python run_brute_force.py --config brute.json
```

Module form:

```powershell
python -m geometric_v1.brute_force --config brute.json
```

On long brute-force runs, DeepFace comparison is isolated in a persistent worker subprocess. Diffusion still runs in the main brute-force process so the diffusion pipeline cache remains warm, while DeepFace model weights stay loaded across attempts. If the DeepFace worker exits with an error or segmentation fault, that attempt is saved under `failures`, the worker is restarted, and the run can continue or resume.

The `deepface_worker` block controls that subprocess:

- `enabled`: use the persistent worker. Set `false` to fall back to one fresh DeepFace subprocess per attempt.
- `max_attempts_per_worker`: recycle the worker after this many attempts. Lower values are safer; higher values avoid more reloads.
- `timeout_seconds`: maximum time to wait for one DeepFace comparison before marking that attempt as a failure.
- `restart_on_failure`: retry once in a fresh worker if the worker exits or crashes.

`brute.json` controls only:

- which `pipeline.json` to use
- number of random attempts
- brute-force output directory
- success threshold, such as `50.0`
- random seed behavior
- safe resume behavior
- persistent DeepFace worker lifecycle
- parameter ranges for perturbation fields
- whether error/failure runs keep any full image files generated before the error

Each completed `successful` or `unsuccessful` run folder contains:

```text
original.png
perturbed.png
original_diffused.png
perturbed_diffused.png
report.json
sampled_config.json
```

Folder names are deterministic:

```text
output/brute_force/successful/run_000012
output/brute_force/unsuccessful/run_000013
output/brute_force/failures/run_000014
```

Success means the average `match_percent` across enabled DeepFace models that completed without error is less than or equal to `success_threshold`. Completed runs above the threshold are saved under `unsuccessful`. Actual error runs, including pipeline errors or DeepFace model errors, are saved under `failures`. If a DeepFace model errors, that model is not counted in the average and its error remains in `report.json`.

`failure` folders always contain `sampled_config.json` and `report.json`. `save_unsuccessful` applies only to error/failure attempts. When it is `false`, any full image files produced before the error are removed. Threshold failures are always saved in `unsuccessful`.

Resume behavior:

- Set `"resume": true` in `brute.json` to safely continue after an interrupted terminal run.
- A run is complete only when its final folder contains both `report.json` and `sampled_config.json`, and `report.json` contains `brute_force.status`.
- Completed run folders are skipped and never overwritten.
- Missing run numbers are executed.
- Incomplete `successful`, `unsuccessful`, or `failures` run folders are moved aside under `failures/incomplete_*` before that run number is retried.
- Stale temporary `_working/run_000000_*` folders are cleaned before retrying.

Seed behavior:

- `seed` controls deterministic random sampling.
- If `randomize_attempt_seed` is `true`, each attempt gets a random seed from `attempt_seed_range`.
- If `randomize_attempt_seed` is `false`, every attempt uses the fixed `seed` as the pipeline, diffusion, and first perturbation seed.
- Resume preserves the same attempt seed for each run number. If `brute_report.json` has an attempt seed, it is reused; otherwise the seed is regenerated deterministically from `brute.json`.
- Enabled perturbation steps receive seeds starting at the attempt seed, then incrementing by one in pipeline order.

Each run's `sampled_config.json` is a runnable copy of `pipeline.json` with sampled perturbation values inserted. It also resolves the input and output paths to absolute paths so it can run correctly from inside the run folder.

## Batch Brute Force

`batch_brute.json` runs brute force for every image and prompt combination. It does not duplicate brute-force logic; it writes combo-local configs and calls the normal brute-force runner.

Run:

```powershell
python run_batch_brute_force.py --config batch_brute.json
```

Module form:

```powershell
python -m geometric_v1.batch_brute_force --config batch_brute.json
```

`batch_brute.json` controls:

- `brute_config`: path to `brute.json`
- `pipeline_config`: optional override; omitted means use the one inside `brute.json`
- `images_dir`
- `image_extensions`, defaulting to `.png`, `.jpg`, `.jpeg`, `.webp`
- `recursive`
- `prompts`
- `output_dir`
- `skip_existing`
- `overwrite_existing`
- `parallel_combinations`

`pipeline.json` remains the source of truth for diffusion settings, CPU/GPU flag, DeepFace enabled models, and enabled perturbation methods. `brute.json` remains the source of truth for trials, success threshold, seed behavior, `attempt_seed_range`, `save_unsuccessful`, and perturbation parameter ranges. Batch mode overrides only the input image, prompt, and output folder for each image/prompt combo.

Output layout:

```text
output/batch_brute/
  batch_report.json
  image_000001_<image_stem>/
    prompt_000000_<short_hash>/
      successful/
      unsuccessful/
      failures/
      brute_report.json
```

`batch_report.json` summarizes total images, total prompts, total planned brute attempts, per-combo status, prompt text, image path, success/unsuccess/failure counts, each combo's `brute_report.json`, and elapsed time.
Combo statuses are `completed`, `skipped`, `resumed`, or `failed`.

Safe batch reruns:

- A combo is complete only when its `brute_report.json` has `summary` and exactly `brute.trials` attempts.
- If `skip_existing` is `true`, only complete combos are skipped.
- If a combo is incomplete, batch mode rewrites the combo-local configs and reruns plain brute force with resume enabled.
- If `skip_existing` is `false`, complete combos are not overwritten unless `overwrite_existing` is `true`.
- After terminal interruption, rerun the same batch command. Complete combos are skipped, incomplete combos are resumed, and missing combos are run.

If `parallel_combinations` is greater than `1`, multiple image/prompt combos run at the same time with a bounded process pool. Be careful with values above `1`: each combo can run diffusion on the GPU, so CUDA memory pressure can rise quickly.

When `randomize_attempt_seed` is `true` in `brute.json`, batch mode pre-generates unique attempt seeds across the whole batch. For example, `2 images x 2 prompts x 100 trials` produces `400` distinct attempt seeds from `attempt_seed_range`.

## Loss-Guided Optimization

`loss.json` runs a black-box, loss-guided optimizer for one image and one prompt. It is designed for the case where you want the perturbed image to still match the original before diffusion, but to match less after diffusion.

Run:

```powershell
python run_loss_pipeline.py --config loss.json
```

Module form:

```powershell
python -m geometric_v1.loss_pipeline --config loss.json
```

This is not true end-to-end backpropagation. Current Flux, DeepFace, and the geometric perturbation stack are treated as a black box. The v1 optimizer evaluates candidate perturbation parameters, measures the loss, and updates the next candidate with SPSA by default.

The main terms are:

- `alpha_pre`: identity match percentage between `original.png` and `perturbed.png`. This should stay high so the pre-diffusion perturbation does not simply destroy the person.
- `alpha_post`: identity match percentage between cached `original_diffused.png` and each iteration's `perturbed_diffused.png`. This is the term the optimizer tries to reduce.
- `beta`: visual similarity constraints between `original.png` and `perturbed.png`, such as PSNR, SSIM, optional single-image FID, and optional LPIPS.

The default objective is:

```text
loss =
  w_alpha_post * alpha_post
  + w_alpha_pre * max(0, alpha_pre_target - alpha_pre)^2
  + w_psnr * max(0, psnr_target - psnr)^2
  + w_ssim * max(0, ssim_target - ssim)^2
  + w_fid * max(0, fid - fid_target)^2
  + parameter regularization
```

Every component can be enabled, disabled, or reweighted from `loss.json`. `alpha_post` is enabled by default because it is the actual attack objective. `alpha_pre` and beta constraints are optional guardrails.

Inside `objective.beta`, the explicit switches `use_psnr`, `use_ssim`, `use_fid`, and `use_lpips` decide whether each beta metric participates. The per-metric `enabled` field is also accepted, but the `use_*` switches are the easiest way to turn terms on and off. LPIPS is not computed unless `use_lpips` is `true`.

Optimizers:

- `spsa`: default black-box optimizer. Each iteration evaluates plus/minus perturbations and estimates a gradient without backpropagating through Flux or DeepFace.
- `random_search`: random candidates within the configured bounds.
- `differential_evolution`: SciPy's differential evolution optimizer. This is optional and can be expensive because each candidate runs diffusion and DeepFace.

Output layout:

```text
output/loss_run_<timestamp>/
  loss_config.json
  original.png
  original_diffused.png
  best/
    perturbed.png
    perturbed_diffused.png
    report.json
  iterations/
    iter_000001/
      perturbed.png
      perturbed_diffused.png
      metrics.json
  loss_history.json
  report.json
```

`original_diffused.png` is generated once and cached, because the original image and prompt do not change. Each candidate iteration generates only a new `perturbed.png`, a new `perturbed_diffused.png`, alpha/beta metrics, and a loss score.

FID note: the built-in FID option is intentionally marked weak for single-image use. Real FID is a distribution metric over many images and usually Inception features. The v1 implementation uses a simple RGB Gaussian distance so you can use it as a rough optional constraint, not as a paper-grade score. LPIPS is also optional and only runs if the `lpips` Python package is installed.

Ubuntu A6000 installs do not need a script change for the default `loss.json` because PSNR, SSIM, and the optional lightweight FID approximation use existing dependencies. If you turn on `objective.beta.use_lpips`, install the optional LPIPS package with `INSTALL_LPIPS=1 bash linux-gpu/install_linux_a6000.sh`.

For a deeper explanation of alpha/beta terms, SPSA, black-box optimization, and how to interpret the reports, see `concepts.md`.

## Sample Configs

Explanatory examples live in `sample_jsons/`:

```text
sample_jsons/sample_pipeline.json
sample_jsons/sample_brute.json
sample_jsons/sample_batch_brute.json
```

## Config Shape

```json
{
  "input": "samples/original.png",
  "output_dir": "output/run_001",
  "prompt": "make the person look like a professional studio portrait",
  "seed": 7,
  "perturbations": [
    {"method": "homography", "enabled": true, "strength": 0.04},
    {"method": "thin-plate-spline", "enabled": true, "strength": 0.025, "grid": 7},
    {"method": "delaunay", "enabled": true, "strength": 0.025, "grid": 8},
    {"method": "fft-phase", "enabled": true, "strength": 0.20, "coefficients": 10},
    {"method": "elastic", "enabled": true, "strength": 0.025, "sigma": 12.0},
    {
      "method": "rolling-shutter",
      "enabled": true,
      "strength": 0.03,
      "rolling_frequency": 1.5,
      "rolling_shear": 0.02
    }
  ],
  "diffusion": {
    "cpu": false,
    "device": "auto",
    "gpu_index": 0,
    "models": {
      "instruct_pix2pix": {
        "enabled": false,
        "model_id": "timbrooks/instruct-pix2pix",
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
        "image_guidance_scale": 1.0,
        "max_size": 512,
        "seed": 7
      },
      "flux2_klein": {
        "enabled": true,
        "model_id": "black-forest-labs/FLUX.2-klein-4B",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "max_size": 768,
        "height": null,
        "width": null,
        "max_sequence_length": 512,
        "text_encoder_out_layers": [9, 18, 27],
        "torch_dtype": "bfloat16",
        "cpu_offload": false,
        "sigmas": null,
        "seed": 7
      }
    }
  },
  "deepface": {
    "enabled": true,
    "detector_backend": "skip",
    "distance_metric": "cosine",
    "enforce_detection": false,
    "align": false,
    "workers": "auto",
    "models": {
      "SFace": true,
      "OpenFace": true,
      "Facenet": true,
      "Facenet512": true
    }
  }
}
```

Diffusion model selection:

- `pipeline.json` contains two diffusion model blocks: `instruct_pix2pix` and `flux2_klein`.
- Each model block has an `enabled` boolean.
- If only one model is enabled, that model runs.
- If both are enabled, `flux2_klein` runs and `instruct_pix2pix` is ignored.
- The pipeline never runs both diffusion models in the same run.
- Pipeline, diffuse-only, brute-force, and batch brute-force reports write `diffusion.used_model`, `diffusion.used_model_id`, `diffusion.selected_model`, and `diffusion.selected_model_id`.

FLUX.2 Klein tunable options:

- `num_inference_steps`: denoising steps. The model card example uses `4`.
- `guidance_scale`: prompt guidance. The model card example uses `1.0`.
- `max_size`: used by this project to resize the input image when `height` and `width` are not set.
- `height` and `width`: optional explicit output canvas size, rounded down to multiples of 8.
- `max_sequence_length`: prompt token budget passed to Diffusers.
- `text_encoder_out_layers`: encoder layers used by the Flux2Klein pipeline.
- `torch_dtype`: `bfloat16`, `float16`, or `float32`; CUDA defaults should generally use `bfloat16`.
- `cpu_offload`: enables Diffusers model CPU offload when CUDA is selected.
- `sigmas`: optional custom scheduler sigma list. Leave `null` for normal use.
- `seed`: generator seed.

Diffusion device notes:

- Set `"cpu": true` to force diffusion onto CPU even when CUDA is available.
- Keep `"cpu": false` with `"device": "auto"` to use CUDA when PyTorch can see it, otherwise CPU.
- The pipeline writes `diffusion.resolved_device` into `report.json` so you can confirm the actual device used.
- GPU-mode pipeline runs use batched diffusion for `original.png` and `perturbed.png`. CPU-mode runs keep the old sequential diffusion path.
- If batched GPU diffusion errors, the pipeline falls back to sequential diffusion and records `diffusion.batch_error` in `report.json`.

DeepFace model notes:

- Only `SFace`, `OpenFace`, `Facenet`, and `Facenet512` are supported by this project. Other DeepFace model names in older configs are ignored.
- `workers` can be `"auto"` or an integer. It is only used when the full pipeline resolves diffusion to CUDA. CPU-mode runs and standalone DeepFace commands stay sequential.
- In `"auto"` mode, the worker count is capped at 3 and also limited by enabled model count, CPU count, and available RAM. If you set an integer manually, the runner respects it up to the number of enabled supported models. The chosen value is recorded under `deepface.execution.resolved_workers` in `report.json`.
- The project caches known DeepFace weight files into `~\.deepface\weights` before each model runs. The first run can download the four supported model weights.
- Each enabled model records either a comparison result or an error in `report.json`.

## Independent Commands

Create only `perturbed.png`:

```powershell
python run_perturbations.py --config pipeline.json
```

Module form:

```powershell
python -m geometric_v1.perturbations_cli --config pipeline.json
```

Edit one image with InstructPix2Pix:

```powershell
python run_diffusion.py `
  --image samples\original.png `
  --prompt "make the person look like a professional studio portrait" `
  --output output\single_diffused.png `
  --gpu-index 0
```

Force the standalone diffusion command onto CPU:

```powershell
python run_diffusion.py `
  --image samples\original.png `
  --prompt "make the person look like a professional studio portrait" `
  --output output\single_diffused_cpu.png `
  --cpu
```

Compare two images with DeepFace:

```powershell
python run_deepface_compare.py `
  --image-a output\run_001\original_diffused.png `
  --image-b output\run_001\perturbed_diffused.png `
  --output output\run_001\deepface_only_report.json `
  --config pipeline.json
```

Check DeepFace model loading on generated sample images:

```powershell
python run_deepface_model_check.py --output output\deepface_model_check.json
```

The local supported-model check covers:

```text
SFace, OpenFace, Facenet, Facenet512
```

## Match Percentage

DeepFace returns a distance and threshold per model. This project reports:

```text
match_percent = max(0, min(100, 100 * (1 - distance / (2 * threshold))))
```

That makes `100%` mean identical embedding distance, about `50%` mean the model threshold boundary, and `0%` mean very far beyond the threshold.

## References

- Hugging Face diffusers InstructPix2Pix docs: <https://huggingface.co/docs/diffusers/api/pipelines/pix2pix>
- FLUX.2 Klein model card: <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- Hugging Face Diffusers Flux2 docs: <https://huggingface.co/docs/diffusers/main/api/pipelines/flux2>
- DeepFace verify/model docs: <https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py>
- TensorFlow pip install notes: <https://www.tensorflow.org/install/pip>
- PyTorch install selector: <https://pytorch.org/get-started/>
