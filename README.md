# Geometric V1

`geometric-v1` is a full image-to-report pipeline:

1. Read one input image and a text prompt from `pipeline.json`.
2. Build `perturbed.png` by applying enabled geometric/frequency perturbations in order.
3. Send both `original.png` and `perturbed.png` through InstructPix2Pix with the same prompt.
4. Compare `original_diffused.png` and `perturbed_diffused.png` with enabled DeepFace models.
5. Write all four images plus `report.json` to the configured output folder.

## Setup

Python 3.11 is recommended for this project because it is a stable intersection for PyTorch, diffusers, TensorFlow, and DeepFace on Windows.

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

This project uses TensorFlow 2.12.1 because the DeepFace model named `DeepFace` needs `LocallyConnected2D`, which is missing from newer TensorFlow releases. PyTorch needs a newer `typing-extensions`, so the final `typing-extensions` command is intentional even though TensorFlow's package metadata asks for an older version. This exact combination was tested locally with CUDA PyTorch and all DeepFace recognition models.

If PyTorch is not already installed in your environment, install it with the selector at the official PyTorch install page after `requirements.txt`, then rerun:

```powershell
python -m pip install "typing-extensions>=4.14,<5"
```

## Project Layout

The package is directly at the repository root:

```text
geometric_v1/
run_pipeline.py
run_perturbations.py
run_diffusion.py
run_deepface_compare.py
run_deepface_model_check.py
pipeline.json
```

There is no `src/` layout.

## Full Pipeline

Edit `pipeline.json`, especially:

- `input`
- `output_dir`
- `prompt`
- perturbation strengths
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

DeepFace model notes:

- All DeepFace recognition model booleans are enabled by default because this environment was set up and tested with all of them.
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
