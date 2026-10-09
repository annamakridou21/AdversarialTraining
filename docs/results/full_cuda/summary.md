# CIFAR-10 robustness evaluation

Measured on the test split. Robust accuracy requires clean and attacked correctness.

| Model | Budget | Samples | Clean (%) | Robust (%) | ASR (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | 2/255 | 10000 | 78.70 | 21.69 | 72.44 |
| Standard | 4/255 | 10000 | 78.70 | 1.14 | 98.55 |
| Standard | 6/255 | 10000 | 78.70 | 0.05 | 99.94 |
| Standard | 8/255 | 10000 | 78.70 | 0.00 | 100.00 |
| Adversarial | 2/255 | 10000 | 70.59 | 54.63 | 22.61 |
| Adversarial | 4/255 | 10000 | 70.59 | 35.82 | 49.26 |
| Adversarial | 6/255 | 10000 | 70.59 | 20.50 | 70.96 |
| Adversarial | 8/255 | 10000 | 70.59 | 10.42 | 85.24 |

## Checkpoint provenance

- Standard: selected epoch 4, SHA-256 `5e61a0f146cca75d040b21e0b7556d514625553b10cee97bda1e127b7ba1c709`.
  Stage training budget: 5 epoch(s); samples: {'train': 45000, 'validation': 5000, 'test': 10000}.
- Adversarial: selected epoch 4, SHA-256 `1204386dcfcf2d81e610604387c1586ce59dc001bdf0d7d53c4b71c4f90ee67e`.
  Stage training budget: 5 epoch(s); samples: {'train': 45000, 'validation': 5000, 'test': 10000}.

PGD settings: 20 steps, 5 restarts, step size 2/255; seed 42.

These are empirical attack results, not a certificate of robustness.
