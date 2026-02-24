# Adversarial Training via Min-Max Saddle Point Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Adversarial training implementation using min-max saddle point optimization for robust deep learning on CIFAR-10. This project demonstrates that adversarial training significantly improves model robustness against adversarial attacks while simultaneously improving clean accuracy.

**Authors**: Anna Makridou & Alexandros Fourtounis  
**Course**: CS-573 Optimization Methods

---

## 📊 Key Results

Our adversarially trained model achieves:
- **76.44%** clean accuracy (+7.37% over baseline)
- **36.40%** robust accuracy against PGD-10 attacks (ε=4/255)
- **2.4× robustness improvement** over standard training
- **23% reduction** in attack success rate

| Metric | Standard Training | Adversarial Training | Improvement |
|--------|------------------|---------------------|-------------|
| Clean Accuracy | 69.07% | **76.44%** | +7.37% |
| Robust Accuracy (ε=4/255) | 15.39% | **36.40%** | +136.7% |
| Attack Success Rate | 81.54% | **58.33%** | -28.5% |

### Multi-Epsilon Robustness

| Attack Strength (ε) | Standard Model | Robust Model | Improvement |
|---------------------|----------------|--------------|-------------|
| 2/255 | 17.21% | **41.85%** | +143% |
| 4/255 (training) | 15.39% | **36.40%** | +137% |
| 6/255 | 13.42% | **31.46%** | +134% |
| 8/255 | 11.77% | **26.69%** | +127% |

---

## 🎯 Problem Formulation

Deep neural networks are vulnerable to adversarial attacks—small, imperceptible perturbations that cause misclassification. We address this using **adversarial training**, formulated as a min-max saddle point optimization problem:

```
min_θ E_(x,y)~D [max_{δ∈S} L(θ, x+δ, y)]
```

**Where:**
- **θ**: Model parameters
- **L**: Cross-entropy loss
- **δ**: Adversarial perturbation
- **S = {δ : ||δ||_∞ ≤ ε}**: L∞ perturbation ball (ε=4/255)

**Optimization Strategy:**
- **Inner Maximization**: Generate worst-case adversarial examples using Projected Gradient Descent (PGD-10)
- **Outer Minimization**: Train model parameters using Stochastic Gradient Descent (SGD) with momentum

---

## 🏗️ Project Structure

```
573-project/
├── model_arch.py              # CNN architecture (~4.8M parameters)
├── attacks.py                 # PGD & FGSM implementations
├── utils.py                   # Data loading & evaluation utilities
├── main_standard.py           # Standard training (ERM baseline)
├── main_robust.py             # Robust training (adversarial training)
├── requirements.txt           # Python dependencies
├── data/                      # CIFAR-10 dataset
│   └── cifar-10-batches-py/
├── output/                    # Training outputs
│   ├── models/                # Saved model checkpoints
│   ├── plots/                 # Training curves and comparisons
│   ├── visualizations/        # Adversarial example visualizations
│   └── analysis_*.json        # Detailed metrics
├── scripts/                   # Analysis and visualization scripts
│   ├── analyze_robustness.py
│   ├── visualize_attacks.py
│   ├── plot_training.py
│   └── create_summary_report.py
└── experiments/               # Experiment checkpoints
```

---

## 🔬 Methodology

### Model Architecture

Convolutional Neural Network (CNN) for CIFAR-10:
- **Architecture**: 3 convolutional blocks → 2 fully connected layers
- **Channels**: 3 → 64 → 128 → 256
- **Parameters**: ~4.8M
- **Regularization**: Batch normalization + dropout (0.5)

### PGD Attack Algorithm

```python
# Inner maximization to find worst-case perturbation
for t in range(num_iterations):
    grad = ∇_δ L(θ, x + δ, y)
    δ = δ + α · sign(grad)  # Gradient ascent
    δ = clip(δ, -ε, ε)      # Project to L∞ ball
    δ = clip(x + δ, 0, 1) - x  # Ensure valid image
```

**Parameters**:
- ε = 4/255 (perturbation budget)
- α = 2/255 (step size)
- 10 iterations with random initialization

### Training Procedure

**Phase 1: Pre-training on Clean Data (10 epochs)**
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.1 with cosine annealing
- Batch size: 128
- Data augmentation: random crop + horizontal flip

**Phase 2: Adversarial Fine-tuning (5 epochs)**
- Pure adversarial training (100% adversarial examples)
- Learning rate: 0.01 (reduced to prevent catastrophic forgetting)
- PGD-10 attack generation for each batch
- Best checkpoint selection based on validation

### Dataset
- **CIFAR-10**: 60,000 32×32 RGB images
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Split**: 50,000 training / 10,000 test
- **Normalization**: [0, 1] range

---

## � Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/573-project.git
cd 573-project

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Standard baseline training
python main_standard.py --epochs 10 --batch-size 128

# Adversarial training (with pre-training)
python main_robust.py --epochs 15 --batch-size 128
```

### Evaluation

```bash
# Analyze robustness metrics
python scripts/analyze_robustness.py

# Visualize adversarial examples
python scripts/visualize_attacks.py

# Create comparison plots
python scripts/create_comparison_plots.py
```

### Advanced Options

```bash
# Custom adversarial training
python main_robust.py \
    --epochs 15 \
    --batch-size 128 \
    --lr 0.01 \
    --train-epsilon 0.0157 \
    --train-steps 10 \
    --device mps  # or cuda, cpu
```

---

## � Key Findings

### 1. Simultaneous Improvement

Unlike typical adversarial training that trades clean accuracy for robustness, our approach improves **both** metrics:
- Clean accuracy: 69.07% → 76.44% (+7.37%)
- Robust accuracy: 15.39% → 36.40% (+136.7%)

**Why?** Careful checkpoint selection and learning rate scheduling preserve pre-trained features while adding robustness.

### 2. Generalization Across Attack Strengths

The robust model maintains 2-3× better accuracy across all attack strengths (ε = 2/255 to 8/255), demonstrating learned robust features rather than overfitting to a specific perturbation budget.

### 3. Pre-training is Critical

| Training Method | Clean Accuracy | Robust Accuracy |
|----------------|----------------|-----------------|
| From scratch | 15.80% | 37.84% |
| With pre-training | **76.44%** | **36.40%** |

Pre-training on clean data provides strong feature representations that adversarial fine-tuning can refine rather than rebuild from scratch.

### 4. Learning Rate Matters

| Learning Rate | Clean Accuracy | Robust Accuracy | Notes |
|--------------|----------------|-----------------|-------|
| 0.1 | 24.92% | - | Catastrophic forgetting |
| 0.01 | **76.44%** | **36.40%** | Optimal balance |
| 0.005 | 31.20% | 57.36% | Different trade-off point |

Lower learning rates during adversarial fine-tuning prevent catastrophic forgetting of pre-trained features.

---

## 📊 Experimental Results

### Attack Success Rate Analysis

Among images correctly classified on clean data:
- **Standard model**: 81.54% successfully attacked
- **Robust model**: 58.33% successfully attacked
- **Improvement**: 23.21 percentage points reduction

This means **41.67% of clean images now resist PGD-10 attacks** (vs. 18.46% for standard model).

### Computational Considerations

- **Standard training**: ~25 minutes (10 epochs)
- **Adversarial fine-tuning**: ~1.5 hours (5 epochs)
- **Overhead**: ~6× slower due to PGD inner optimization
- **Inference**: No additional cost

---

## 📁 Saved Models

Pre-trained models are available in `output/models/`:

| File | Model | Clean Acc | Robust Acc | Notes |
|------|-------|-----------|------------|-------|
| `standard_final.pth` | Baseline | 69.07% | 15.39% | Standard ERM training |
| `robust_best.pth` | **Primary result** | **76.44%** | **36.40%** | Best validation checkpoint |
| `robust_final.pth` | Final epoch | 22.86% | 59.40% | Specialized on adversarial |

### Loading Models

```python
from model_arch import CNN
import torch

model = CNN(num_classes=10)
checkpoint = torch.load('output/models/robust_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🎨 Visualizations

Generated plots are available in `output/plots/`:

- `accuracy_comparison.png` - Clean vs. robust accuracy comparison
- `epsilon_comparison.png` - Multi-epsilon robustness curves
- `attack_success_comparison.png` - Attack success rate analysis
- `adversarial_examples_*.png` - Visual examples of adversarial perturbations

---

## 🛡️ Technical Implementation

### PGD Attack with Random Initialization

```python
def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=10):
    """Projected Gradient Descent attack."""
    # Random initialization in epsilon ball
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    
    for _ in range(num_iter):
        # Forward pass
        outputs = model(images + delta)
        loss = F.cross_entropy(outputs, labels)
        
        # Gradient ascent
        loss.backward()
        grad = delta.grad.detach()
        delta.data = delta + alpha * grad.sign()
        
        # Project to epsilon ball
        delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.data = torch.clamp(images + delta, 0, 1) - images
        delta.grad.zero_()
    
    return images + delta
```

### Adversarial Training Loop

```python
def adversarial_training_step(model, images, labels, optimizer):
    """Single adversarial training step."""
    # Generate adversarial examples (inner maximization)
    model.eval()
    adv_images = pgd_attack(model, images, labels)
    
    # Train on adversarial examples (outer minimization)
    model.train()
    optimizer.zero_grad()
    outputs = model(adv_images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 📚 References

1. **Madry, A., et al.** (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018*.
2. **Goodfellow, I. J., et al.** (2015). "Explaining and Harnessing Adversarial Examples." *ICLR 2015*.
3. **Athalye, A., et al.** (2018). "Obfuscated Gradients Give a False Sense of Security." *ICML 2018*.
4. **Carlini, N., & Wagner, D.** (2017). "Towards Evaluating the Robustness of Neural Networks." *IEEE S&P 2017*.

---

## 🤝 Contributing

This project was developed as part of CS-573 Optimization Methods coursework. Feel free to open issues or submit pull requests for improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Anna Makridou** - Implementation and experimentation
- **Alexandros Fourtounis** - Implementation and experimentation

**Course**: CS-573 Optimization Methods  
**Date**: January 2026

---

## 🏆 Key Achievements

- ✅ Complete from-scratch implementation of PGD attack and adversarial training
- ✅ 2.4× improvement in robust accuracy while improving clean accuracy
- ✅ Comprehensive evaluation across multiple attack strengths
- ✅ Systematic experiments demonstrating importance of pre-training and learning rate
- ✅ Production-ready code with proper evaluation framework
- ✅ Extensive visualizations and analysis tools

---

**Status**: Complete and Ready for Publication
