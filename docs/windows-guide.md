# Windows CUDA setup and reproduction

The full experiment has completed on Windows with an NVIDIA GeForce RTX 4060.
The tested environment used Python 3.13.14, PyTorch 2.11.0+cu128, and torchvision
0.26.0+cu128. All 18 regression tests passed. No separate CUDA toolkit was needed
for the installed PyTorch wheels; a compatible NVIDIA driver is required.

## Open the project

Open PowerShell in the project directory. For a fresh copy, clone the repository
or extract the source archive to a directory of your choice. A virtual
environment, dataset, and model checkpoints are not part of the source archive.

## Create a fresh environment

Use a 64-bit Python 3.13 installation for the recorded dependency lock. Check
`python --version` and `nvidia-smi`, then run from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-windows.txt
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

Installing the CUDA pair first allows the frozen requirements file to reuse
those installed versions. The general `requirements.txt` is less restrictive
and can be used with another compatible Python/platform environment.

The [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v2110)
document the CUDA package pair. Calling `.venv`'s Python directly avoids needing
to activate PowerShell scripts. If an agent sandbox denies temporary file writes,
run these same setup/test commands in normal PowerShell.

## Verify

```powershell
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.verify_release
.\.venv\Scripts\python.exe -m scripts.benchmark_training --device cuda
```

The recorded synthetic benchmark measured 0.043 seconds per clean training batch
and 0.355 seconds per PGD-10 mixed training batch. Full evaluation and loading
are excluded from those measurements.

## Reproduce in a fresh directory

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment --profile full --output-dir output/experiments/reproduction --device cuda --num-workers 0 --seeds 42 --execute
```

CIFAR-10 downloads automatically if it is not already in `data/`. The original
170 MB archive in the current project was verified against its official MD5.
Keep the machine awake and the foreground training terminal open. Progress,
recovery, and optional additional seeds are described in [run-guide.md](run-guide.md).

The original local run and checkpoints are backed up separately; the portable
reference results are in [docs/results/full_cuda](results/full_cuda/README.md).
