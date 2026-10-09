# Reproduce the completed experiment

The full CUDA seed 42 study is complete. Read the [results](results/full_cuda/README.md)
and [report](../report.pdf) without rerunning training. The commands below create
a new reproduction or resume one that you started.

## Windows CUDA

Follow [windows-guide.md](windows-guide.md) for the tested environment. From the
project root, use a fresh output directory:

```powershell
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.run_experiment --profile full --output-dir output/experiments/reproduction --device cuda --num-workers 0 --seeds 42 --execute
```

Omit `--execute` to preview the complete plan. Keep the computer awake while it
runs. The measured study was run from the Desktop project folder; all commands
use paths relative to the project root so the folder can be moved.

## Other platforms

Install a compatible PyTorch/torchvision pair and `requirements.txt`, then:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=4 python -m scripts.run_experiment --profile full --output-dir output/experiments/reproduction --device mps --num-workers 0 --seeds 42 --execute
```

Use `cuda` on a CUDA device or `cpu` for CPU execution. On macOS, `caffeinate -i`
can precede the Python command to prevent idle sleep. Cross-platform results
need not be bitwise identical to the recorded Windows run.

## Protocol

1. Ten clean pre-training epochs, initial learning rate 0.01, decay after epochs 6 and 9.
2. Five clean continuation epochs and five mixed adversarial epochs from the same final pre-training weights.
3. Both branches start at learning rate 0.001, decayed after epochs 3 and 4; each receives 1,760 updates.
4. Mixed loss weight 0.5, PGD-10 training, epsilon 4/255, step size 2/255.
5. Validation selects checkpoints. Final PGD-20 evaluation uses five restarts at budgets 2/255, 4/255, 6/255, and 8/255 on all 10,000 test images.
6. Raw metrics, a comparison plot, and a Markdown summary are generated.

The split is 45,000 training / 5,000 validation images, with batch size 128.
The full protocol is [experiments/full.json](../experiments/full.json).

## Progress and recovery

```powershell
Get-Content output\experiments\reproduction\status.json
Get-Content output\experiments\reproduction\seed_42\logs\pretrain.log -Wait -Tail 10
```

The status file names the current log. Later stages use `clean_control.log`,
`adversarial.log`, `evaluate_clean_control.log`, and `evaluate_adversarial.log`.
Stopping the log viewer does not stop training. In a foreground training terminal,
Ctrl-C interrupts training. Rerun the exact experiment command to resume completed
epochs. The interrupted epoch restarts; analysis restarts from the beginning.

Keep source files, protocol, device, versions, and run settings unchanged when
resuming. The runner rejects mismatched manifests and concurrent writers. If no
first-epoch checkpoint exists, use a fresh output directory. The original
`full_cuda` directory is a completed record; use `reproduction` for another run.

## Artifacts

Within `output/experiments/reproduction/`, `experiment.json` records launch
settings and source hashes, and `status.json` records completion or failure.
`seed_42/` contains the three training stages, `logs/`, `analysis/`, `summary.md`,
and `comparison.png`. A partial directory is not a completed comparison.

The submitted compact records are under `docs/results/full_cuda/`. They include
portable paths, exact counts, full histories, checkpoint hashes, and the original
training-time source manifest. `python -m scripts.verify_release` verifies them.

## Optional extensions

Additional seeds measure variability beyond this completed study. To run only
new seeds without repeating seed 42:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_experiment --profile full --output-dir output/experiments/additional_seeds --device cuda --num-workers 0 --seeds 43 44 --execute
```

This produces separate summaries, not an automatic aggregate or significance
test. Independent attack evaluations and loss-weight sweeps are further studies.
