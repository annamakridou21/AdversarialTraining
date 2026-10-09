# Completed CIFAR-10 CUDA study

Full protocol, seed 42, completed on 9 October 2026. The figures and metrics
represent all 10,000 official test images. Training used 45,000 images and
validation used 5,000. Both branches share the same final pre-training checkpoint.

| File | Contents |
| --- | --- |
| `clean_control.json`, `adversarial.json` | Exact test counts, PGD settings, checkpoint metadata and hashes |
| `pretrain_history.json` | Ten pre-training epochs and configuration |
| `clean_control_history.json`, `adversarial_history.json` | Five branch epochs, learning rates, validation, and durations |
| `protocol.json` | Full experimental settings |
| `experiment.json` | Training-time source/protocol hashes and launch settings |
| `checkpoint_hashes.json` | Selected checkpoint and shared initialization hashes |
| `summary.md`, `comparison.png` | Generated report and plot |
| `status.json` | Completion marker |
| `checksums.sha256` | SHA-256 checksums for the exported files |

The export replaces local absolute paths with repository-relative paths and
normalizes Windows separators. Numerical values, histories, and stored hashes
are preserved. `experiment.json` describes files present when training launched;
the later export, packaging, and verification utilities were not training inputs.

Checkpoint files remain under `output/experiments/full_cuda/seed_42/`. A separate
local checkpoint archive is produced under `output/release/`; source ZIPs do not
include model binaries, datasets, environments, or the historical report.

Regenerate the plot and summary from these JSON files using the commands in the
root README. Regenerate the export from the original local run with
`python -m scripts.export_results`. Verify checksums, metrics, provenance, and
documentation links with `python -m scripts.verify_release`.

The study covers one seed and one attack family. It does not estimate variation
across training runs or certify robustness. The earlier subset pilot is stored
separately in `../pilot/`.
