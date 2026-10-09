# Validation record

Completed study: 9 October 2026, seed 42, Windows / NVIDIA RTX 4060.

## Verified outcome

The full run completed clean pre-training, a matched clean continuation, mixed
adversarial fine-tuning, both full test analyses, and figure/report generation.
Its completion marker and portable evidence are in [results/full_cuda/](results/full_cuda/).
The [report](../report.pdf) and [source](../main.tex) describe those measurements.

All 18 regression tests pass in the Windows Python 3.13.14 environment. CUDA
was verified with PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. The exact
installed dependency versions are in [requirements-lock-windows.txt](../requirements-lock-windows.txt).

## Corrections from the historical implementation

| Historical issue | Current behavior |
| --- | --- |
| Normalized tensors clipped as if they were pixels | Raw `[0,1]` pixels; normalization inside the model |
| Attacks modified parameter gradients and model modes | Input-only gradients and restored module modes |
| Only final PGD candidate retained | Training selects highest loss; evaluation retains successful candidates across steps and restarts |
| Incorrect architecture and parameter count | Four convolutions, two pooling stages, 9,356,554 parameters |
| Test data used for selection | Seeded validation split selects checkpoints; test data used in final analysis |
| Uneven final batches misweighted | Metrics and losses weighted by sample count |
| Missing pre-training silently changed initialization | Explicit failure on missing/incompatible checkpoints |
| Best checkpoint missing on short or zero-accuracy runs | Valid initial best score and final-epoch validation |
| Plots used hard-coded values or incompatible inputs | Figures and summaries read matching saved measurements |
| Interrupted writes and concurrent runners could corrupt output | Atomic writes and platform file locks |
| Interrupted training lost optimization state | Epoch-boundary restoration of optimizer, scheduler, random states, and data-loader generators |
| Report implied ACM publication and unverified experiments | Course-project attribution and corrected measured comparison |

The original report and supplied LaTeX source are preserved locally as
`output/archive/report-original.pdf` and `output/archive/main-original.tex`.
The current `report.pdf` replaces them as the project report. No historical
performance claim is used to support the corrected results.

## Automated checks

```bash
python -m unittest discover -s tests -v
python -m scripts.verify_release
```

The 18 regression tests cover PGD/FGSM bounds, zero epsilon, independent gradient
and PGD reference calculations, exact FGSM behavior, model modes, untouched
parameter gradients, invalid inputs, weighted metrics, undefined ASR, training
updates, architecture size, checkpoint round trips, rejection of legacy
checkpoints, data splits, the training/analysis/plot workflow, atomic writes,
best-checkpoint recovery, exact CPU resume, and experiment locking. Synthetic
test fixtures verify execution, not CIFAR-10 accuracy.

The Windows sandbox initially blocked Python temporary-directory writes. Running
the same suite outside that sandbox passed. The full CUDA run also ran outside
the sandbox, with zero data-loader workers. This is an execution-environment
restriction, not a failing project test.

The release verifier checks exact metric counts, selected epochs, shared initial
weights, exported checksums, training-time source hashes, and local Markdown and
LaTeX references. The original local export additionally checks the model binary
hashes. The final report is compiled and its rendered pages are visually checked.
The Linux/Windows GitHub Actions workflow is included; a remote CI run has not
been observed from this transfer copy.

## Experimental evidence

Training used 45,000 examples, validation used 5,000, and each final analysis
used 10,000 test examples. Both branches start from the same final pre-training
checkpoint and receive five epochs (1,760 updates). Both selected checkpoints
are from branch epoch 4. Their selection criteria differ: clean accuracy for
the control and PGD-10 robust accuracy for the adversarial branch.

At epsilon 4/255, PGD-20 with five restarts measured 1.14% robust accuracy for the
control and 35.82% for mixed training. Clean accuracy was 78.70% and 70.59%.
The observed improvement therefore includes a clean-accuracy cost.

The measured synthetic CUDA benchmark was 0.043 seconds per clean training batch
and 0.355 seconds per PGD-10 mixed training batch at batch size 128. These timings
exclude data loading and full evaluation and are not full experiment durations.

The earlier CPU smoke tests and MPS subset pilot are recorded in the transferred
history. Their pilot artifacts remain in [results/pilot/](results/pilot/), and
are not combined with the full CUDA results.

## Scope of completion

This is a completed, documented single-seed coursework study. Repeated seeds,
independent attack suites, parameter sweeps, and formal robustness guarantees are
outside the completed protocol. The findings do not establish statistical
significance, optimal settings, or deployment safety. Authors' individual roles
are not inferred. No software license or publication status is fabricated.
