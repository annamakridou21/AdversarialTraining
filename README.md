# Adversarial Training via Min-Max Saddle Point Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of adversarial training using min-max saddle point optimization for robust deep learning on CIFAR-10.

## 📊 Results

| Metric | Standard Training | Adversarial Training | Improvement |
|--------|------------------|---------------------|-------------|
| Clean Accuracy | 69.07% | **76.44%** | +7.37% |
| Robust Accuracy (ε=4/255) | 15.39% | **36.40%** | +136.7% |
| Attack Success Rate | 81.54% | **58.33%** | -28.5% |

**Key Achievement**: 2.4× robustness improvement while simultaneously improving clean accuracy.

## 🎯 Problem Statement

We address adversarial vulnerability through min-max optimization:

```
min_θ E_(x,y)~D [max_{δ∈S} L(θ, x+δ, y)]
```

- **Inner max**: PGD-10 generates worst-case perturbations (ε=4/255)
- **Outer min**: SGD trains robust model parameters

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/annamakridou21/adversarial-training-minmax.git
cd adversarial-training-minmax
pip install -r requirements.txt
```

### Training

```bash
# Standard baseline
python main_standard.py --epochs 10 --batch-size 128

# Adversarial training
python main_robust.py --epochs 15 --batch-size 128
```

### Evaluation

```bash
python scripts/analyze_robustness.py
python scripts/visualize_attacks.py
```

## 📁 Project Structure

```
├── model_arch.py          # CNN architecture (~4.8M parameters)
├── attacks.py             # PGD & FGSM implementations
├── utils.py               # Data loading & evaluation
├── main_standard.py       # Standard training
├── main_robust.py         # Adversarial training
├── output/                # Models, plots, visualizations
└── scripts/               # Analysis scripts
```

## 🔬 Methodology

**Model**: CNN with 3 conv blocks (64→128→256 channels) + 2 FC layers

**Training**:
1. Pre-train on clean data (10 epochs, LR=0.1)
2. Adversarial fine-tuning (5 epochs, LR=0.01)

**PGD Attack**: ε=4/255, α=2/255, 10 iterations with random initialization

## 💡 Key Findings

1. **Simultaneous improvement** in both clean (+7.37%) and robust (+136.7%) accuracy
2. **Generalizes well** across attack strengths (ε = 2/255 to 8/255)
3. **Pre-training is critical**: Without it, clean accuracy drops to 15.80%
4. **Learning rate matters**: LR=0.01 optimal for adversarial fine-tuning

## 📚 References

1. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR*.
2. Goodfellow, I. J., et al. (2015). "Explaining and Harnessing Adversarial Examples." *ICLR*.

## 📄 License

MIT License - See LICENSE file for details.
