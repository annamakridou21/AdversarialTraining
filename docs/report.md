# Adversarial Training via Min-Max Optimization

Anna Makridou and Alexandros Fourtounis  
CS-573: Optimization Methods, University of Crete  
Corrected course report, 9 October 2026

The complete report is [report.pdf](../report.pdf), built from [main.tex](../main.tex).
This companion note summarizes the verified experiment and links to its evidence.

## Completed experiment

Seed 42 determines the split and initialization. CIFAR-10 is split into 45,000
training images, 5,000 validation images, and all 10,000 official test images.
The CNN has four convolutions, two pooling stages, and 9,356,554 parameters.
Pixels stay in `[0,1]`, with normalization inside the model.

Clean pre-training runs for ten epochs, starting at learning rate 0.01 and
decaying by 0.1 after epochs 6 and 9. Both branches initialize from its final
checkpoint and train for five epochs with fresh optimizers, initial learning
rate 0.001, and decay after epochs 3 and 4. SGD uses momentum 0.9, weight decay
0.0005, and batch size 128. The branches each receive 1,760 updates; adversarial
training uses more computation per update.

The adversarial objective weights clean and adversarial losses equally on the
same batch. Training PGD uses ten steps, one random start, epsilon 4/255, and step
size 2/255. It retains the highest-loss candidate. Evaluation instead retains
successful attacks across iterations and restarts, breaking ties by loss.

Validation selects the clean control by clean accuracy and the adversarial
branch by PGD-10 robustness with one restart. Both selected checkpoints are from
branch epoch 4; the clean validation score ties at epoch 5, retaining the earlier
checkpoint. Test data are not used for selection.

## Full test results

Final evaluation uses PGD-20, five restarts, and step size 2/255. Robust accuracy
requires correct clean and attacked predictions. ASR is conditional on clean
correctness; its denominators differ between the two models.

| Budget | Control robust | Adversarial robust | Control ASR | Adversarial ASR |
| --- | ---: | ---: | ---: | ---: |
| 2/255 | 21.69% | 54.63% | 72.44% | 22.61% |
| 4/255 | 1.14% | 35.82% | 98.55% | 49.26% |
| 6/255 | 0.05% | 20.50% | 99.94% | 70.96% |
| 8/255 | 0.00% | 10.42% | 100.00% | 85.24% |

Clean accuracy is 78.70% for the control and 70.59% for the adversarial branch.
At 4/255 the robust gain is 34.68 percentage points, with an 8.11-point clean
accuracy cost. These aggregate observations do not identify which individual
images differ between models. No paired per-image analysis was saved.

![Full test comparison](results/full_cuda/comparison.png)

[Raw metrics, histories, protocol, and hashes](results/full_cuda/README.md) support
these values. The full run completed on Windows with an RTX 4060, Python 3.13.14,
PyTorch 2.11.0+cu128, and torchvision 0.26.0+cu128. Exact installed versions are
recorded in [the Windows dependency lock](../requirements-lock-windows.txt).

## Interpretation and limits

The experiment measures the effect of the mixed objective relative to an equally
long clean continuation from the same weights. It does not establish optimal
loss weights, prove that pre-training is necessary, or compare equal computation.
Only seed 42 and one attack family are evaluated. Variation across seeds,
independent attacks, and different attack losses remain outside this study's
scope. The measurements are empirical and do not certify robustness.

The [earlier pilot](results/pilot/) used 4,096 training images and 512 validation
and test images on Apple MPS. It is a workflow study and is not combined with the
full-data measurements. The original course report used an invalid preprocessing
contract for its stated attacks; its historical results are superseded here.

## Engineering checks

All 18 regression tests pass, including attack bounds and gradients, metric
counts, model modes, dataset separation, checkpoint recovery, and exact CPU
resume. The results export additionally verifies source/protocol hashes, the
shared pre-training checkpoint, selected checkpoint hashes, and reported counts.
See [validation.md](validation.md) and [run-guide.md](run-guide.md).

## References

- Madry et al. (2018), [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083).
- Goodfellow et al. (2015), [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572).
- Krizhevsky (2009), [CIFAR-10 dataset and technical report](https://www.cs.toronto.edu/~kriz/cifar.html).
