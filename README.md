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
python -m pip install -r requirements.txt
```

If you want a CUDA-specific PyTorch install, follow the selector at the official PyTorch install page and install that wheel before `requirements.txt`.

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
      "VGG-Face": false,
      "Facenet": false,
      "Facenet512": false,
      "OpenFace": false,
      "DeepFace": false,
      "DeepID": false,
      "ArcFace": false,
      "Dlib": false,
      "SFace": true,
      "GhostFaceNet": false,
      "Buffalo_L": false
    }
  }
}
```

DeepFace model notes:

- `SFace` is enabled by default because it is the most stable Windows baseline in this setup.
- The other model booleans are present and can be turned on, but their weights or optional dependencies may need manual setup.
- `Dlib` is disabled by default because Windows installs often need a local C++ toolchain.
- `Buffalo_L` is disabled by default because it may pull extra InsightFace/ONNX dependencies.
- Turn either one on in `pipeline.json` when your environment supports it.
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
