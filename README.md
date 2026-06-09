# Geometric V1

`geometric-v1` is a full image-to-report pipeline:

1. Read one input image and a text prompt from `pipeline.json`.
2. Build `perturbed.png` by applying enabled geometric/frequency perturbations in order.
3. Send both `original.png` and `perturbed.png` through InstructPix2Pix with the same prompt.
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
- `pipeline.json`, `brute.json`, `batch_brute.json`: A6000-oriented configs
- `parameters.md`: explanation of every parameter change

Before installing Python packages on a new Windows laptop, install:

- Python 3.11.x
- Git
- CMake
- Visual Studio 2022 Build Tools with the `Desktop development with C++` workload
- NVIDIA driver and CUDA-compatible PyTorch only if you want GPU diffusion

```powershell
cd path\to\geometric-v1
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$env:CMAKE_ARGS="-DDLIB_USE_CUDA=OFF"
python -m pip install -r requirements.txt
python -m pip install "typing-extensions>=4.14,<5"
```

The `CMAKE_ARGS` line keeps `dlib` on a CPU-only build path. Without it, dlib may try to compile against a partial CUDA toolchain and fail on Windows.

This project pins `numpy>=1.22,<=1.24.3` because TensorFlow 2.12.1 declares that exact compatible range. TensorFlow 2.12.1 is intentional because the DeepFace model named `DeepFace` needs `LocallyConnected2D`, which is missing from newer TensorFlow releases. PyTorch needs a newer `typing-extensions`, so the final `typing-extensions` command is intentional even though TensorFlow's package metadata asks for an older version. This exact combination was tested locally with CUDA PyTorch and all DeepFace recognition models.

The local dashboard adds `fastapi` and `uvicorn[standard]`. These are included in `requirements.txt`; they are only needed for the UI/backend and do not affect the existing CLI entrypoints.

PyTorch is not listed in `requirements.txt` because the right wheel depends on the laptop's CPU/GPU/CUDA setup. If PyTorch is not already installed in your environment, install it with the selector at the official PyTorch install page after `requirements.txt`, then rerun:

```powershell
python -m pip install "typing-extensions>=4.14,<5"
```

For a CPU-only laptop, choose the CPU PyTorch command from the official selector, set `"cpu": true` in `pipeline.json`, and expect diffusion to be slow. For an NVIDIA GPU laptop, choose the CUDA PyTorch command that matches your driver, keep `"cpu": false`, and check `diffusion.resolved_device` in `report.json`.

Verify the install:

```powershell
python -c "import numpy, tensorflow as tf, torch; print('numpy', numpy.__version__); print('tensorflow', tf.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected NumPy on this stack is `1.24.3` or lower within the pinned range. If `dlib` fails during setup, install or repair Visual Studio Build Tools and CMake, reopen PowerShell, set `CMAKE_ARGS` again, and rerun `python -m pip install -r requirements.txt`.

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
run_ui.py
pipeline.json
brute.json
batch_brute.json
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

`brute.json` controls only:

- which `pipeline.json` to use
- number of random attempts
- brute-force output directory
- success threshold, such as `50.0`
- random seed behavior
- safe resume behavior
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
    "model_id": "timbrooks/instruct-pix2pix",
    "cpu": false,
    "device": "auto",
    "gpu_index": 0,
    "num_inference_steps": 10,
    "guidance_scale": 7.5,
    "image_guidance_scale": 1.0,
    "max_size": 512
  },
  "deepface": {
    "enabled": true,
    "detector_backend": "skip",
    "distance_metric": "cosine",
    "enforce_detection": false,
    "align": false,
    "workers": "auto",
    "models": {
      "VGG-Face": true,
      "Facenet": true,
      "Facenet512": true,
      "OpenFace": true,
      "DeepFace": true,
      "DeepID": true,
      "ArcFace": true,
      "Dlib": true,
      "SFace": true,
      "GhostFaceNet": true,
      "Buffalo_L": true
    }
  }
}
```

Diffusion device notes:

- Set `"cpu": true` to force InstructPix2Pix onto CPU even when CUDA is available.
- Keep `"cpu": false` with `"device": "auto"` to use CUDA when PyTorch can see it, otherwise CPU.
- The pipeline writes `diffusion.resolved_device` into `report.json` so you can confirm the actual device used.
- GPU-mode pipeline runs use batched diffusion for `original.png` and `perturbed.png`. CPU-mode runs keep the old sequential diffusion path.
- If batched GPU diffusion errors, the pipeline falls back to sequential diffusion and records `diffusion.batch_error` in `report.json`.

DeepFace model notes:

- All DeepFace recognition model booleans are enabled by default because this environment was set up and tested with all of them.
- `workers` can be `"auto"` or an integer. It is only used when the full pipeline resolves diffusion to CUDA. CPU-mode runs and standalone DeepFace commands stay sequential.
- In `"auto"` mode, the worker count is capped at 3 and also limited by enabled model count, CPU count, and available RAM. The chosen value is recorded under `deepface.execution.resolved_workers` in `report.json`.
- The project caches known DeepFace weight files into `~\.deepface\weights` before each model runs. The first all-model run can download several large files.
- `Dlib` requires a local C++ build toolchain. On this machine it built successfully with `CMAKE_ARGS="-DDLIB_USE_CUDA=OFF"`.
- `Buffalo_L` requires `insightface`, `onnxruntime`, and its Google Drive ONNX weight. These are included in setup and were tested.
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

The local all-model check passed for:

```text
VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L
```

## Match Percentage

DeepFace returns a distance and threshold per model. This project reports:

```text
match_percent = max(0, min(100, 100 * (1 - distance / (2 * threshold))))
```

That makes `100%` mean identical embedding distance, about `50%` mean the model threshold boundary, and `0%` mean very far beyond the threshold.

## References

- Hugging Face diffusers InstructPix2Pix docs: <https://huggingface.co/docs/diffusers/api/pipelines/pix2pix>
- DeepFace verify/model docs: <https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py>
- TensorFlow pip install notes: <https://www.tensorflow.org/install/pip>
- PyTorch install selector: <https://pytorch.org/get-started/>
