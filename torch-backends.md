# PyTorch Attention Backend Testing

These backend switches are useful when Flux diffusion alternates between fast and slow runs with the same images and similar GPU memory use.

Important: PyTorch backend switches are process-local. If you run a Python snippet that sets a backend and exits, the next `python run_batch_brute_force.py --config batch_brute.json` command starts a fresh Python process and loses that setting.

So the backend-setting code must run in the same Python process that imports and runs `geometric_v1`.

## Check Current Backend Flags

Run this inside the active project environment:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
print("cudnn:", torch.backends.cuda.cudnn_sdp_enabled())
print("flash available:", torch.backends.cuda.is_flash_attention_available())
PY
```

## Run Batch Brute With Flash Attention Only

Try this first on an A6000:

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
print("cudnn:", torch.backends.cuda.cudnn_sdp_enabled())

from geometric_v1.batch_brute_force import run_batch_brute_force
run_batch_brute_force(Path("batch_brute.json"))
PY
```

## Run Batch Brute With Memory-Efficient Attention Only

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
print("cudnn:", torch.backends.cuda.cudnn_sdp_enabled())

from geometric_v1.batch_brute_force import run_batch_brute_force
run_batch_brute_force(Path("batch_brute.json"))
PY
```

## Run Batch Brute With cuDNN Attention Only

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(True)

print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
print("cudnn:", torch.backends.cuda.cudnn_sdp_enabled())

from geometric_v1.batch_brute_force import run_batch_brute_force
run_batch_brute_force(Path("batch_brute.json"))
PY
```

## Run Batch Brute With Math Attention Only

This is mostly a correctness or slow-baseline test. It may be much slower.

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
print("cudnn:", torch.backends.cuda.cudnn_sdp_enabled())

from geometric_v1.batch_brute_force import run_batch_brute_force
run_batch_brute_force(Path("batch_brute.json"))
PY
```

## Run Other Entry Points The Same Way

For plain brute force:

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

from geometric_v1.brute_force import run_brute_force
run_brute_force(Path("brute.json"))
PY
```

For a single pipeline run:

```bash
python - <<'PY'
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)

from geometric_v1.pipeline import run_pipeline
run_pipeline(Path("pipeline.json"))
PY
```

## What To Compare

While testing each backend, monitor GPU state in another terminal:

```bash
nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1
```

Compare:

- diffusion seconds per attempt
- peak VRAM
- `clocks.sm`
- `power.draw`
- `temperature.gpu`
- whether any backend crashes with "no available kernel" or similar

If a backend errors, it probably does not support the exact Flux tensor shape or dtype for that run. Move to the next backend.

## Notes

- These settings do not permanently change the environment.
- They affect only the current Python process.
- They should be set before importing/running the pipeline code.
- Keep `parallel_combinations` at `1` while testing GPU diffusion performance.
- If Flash-only is stable and consistently fast, it is probably the best A6000 option.
