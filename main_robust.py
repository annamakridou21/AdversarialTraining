"""Robust Training via Adversarial Training (Min-Max Optimization with Mixed Training).

This script implements adversarial training with MIXED training approach:
    min_θ E_{(x,y)~D} [0.5 * L(θ, x, y) + 0.5 * max_{δ∈S} L(θ, x + δ, y)]

The inner maximization is solved using PGD, and the outer minimization
uses SGD on a mixture of clean and adversarial examples (50-50 mix).
This prevents catastrophic forgetting while maintaining robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

from model_arch import CNN
from utils import get_cifar10_loaders, evaluate_clean, evaluate_adversarial, save_checkpoint, load_checkpoint
from attacks import PGDAttack


def train_robust(model, train_loader, optimizer, criterion, attack, device, epoch, adv_ratio=0.5):
    """Robust training loop (Adversarial Training).
    
    Solves the min-max saddle point problem:
    - Inner max: Generate adversarial examples using PGD
    - Outer min: Update model parameters to minimize loss on adversarial examples
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        attack: PGDAttack object for inner maximization
        device: Device to train on
        epoch: Current epoch number
        adv_ratio: Ratio of adversarial examples in mixed training (default 0.5 for 50-50)
        
    Returns:
        avg_loss: Average training loss (on adversarial examples)
        accuracy: Training accuracy (on adversarial examples)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Robust]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # ============================================================
        # INNER MAXIMIZATION: Generate adversarial examples
        # Solve: max_{δ∈S} L(θ, x + δ, y) using PGD
        # ============================================================
        model.eval()  # Set to eval mode for attack generation
        images_adv = attack(images, labels, random_start=True)
        
        # ============================================================
        # OUTER MINIMIZATION: Update model on MIXED examples
        # MIXED TRAINING: 50% clean + 50% adversarial
        # This prevents catastrophic forgetting of clean features
        # ============================================================
        model.train()  # Set back to train mode
        optimizer.zero_grad()
        
        # Loss on adversarial examples
        outputs_adv = model(images_adv)
        loss_adv = criterion(outputs_adv, labels)
        
        # Loss on clean examples
        outputs_clean = model(images)
        loss_clean = criterion(outputs_clean, labels)
        
        # Combined loss: adv_ratio * adversarial + (1-adv_ratio) * clean
        loss = adv_ratio * loss_adv + (1 - adv_ratio) * loss_clean
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Statistics (use adversarial examples for accuracy tracking)
        running_loss += loss.item()
        _, predicted = outputs_adv.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main(args):
    # Set device (prioritize MPS for Apple Silicon, then CUDA, then CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Initialize model
    print("Initializing model...")
    model = CNN(num_classes=10, dropout_rate=args.dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # PRE-TRAINING TRICK: Load standard model weights if available
    start_epoch = 1
    if args.pretrain_path:
        print(f"\n{'='*60}")
        print("PRE-TRAINING TRICK: Loading Standard Model Weights")
        print(f"{'='*60}")
        print(f"Loading from: {args.pretrain_path}")
        try:
            checkpoint = torch.load(args.pretrain_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate pre-trained model
            pretrain_clean_acc, _ = evaluate_clean(model, test_loader, device)
            print(f"Pre-trained Clean Accuracy: {pretrain_clean_acc:.2f}%")
            print("✓ Successfully loaded standard model weights")
            print("  This gives the robust model a strong starting point!")
            print(f"{'='*60}\n")
        except FileNotFoundError:
            print(f"⚠ Warning: Pre-training file not found: {args.pretrain_path}")
            print("  Starting from random initialization instead.\n")
        except Exception as e:
            print(f"⚠ Warning: Could not load pre-training weights: {e}")
            print("  Starting from random initialization instead.\n")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma
    )
    
    # Initialize PGD attack for training
    print(f"\nPGD Attack Configuration:")
    print(f"  Epsilon: {args.train_epsilon:.4f} ({args.train_epsilon*255:.1f}/255)")
    print(f"  Alpha: {args.train_alpha:.4f}")
    print(f"  Iterations: {args.train_steps}")
    
    train_attack = PGDAttack(
        model=model,
        epsilon=args.train_epsilon,
        alpha=args.train_alpha,
        num_iter=args.train_steps,
        device=device
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Robust Training (Min-Max Saddle Point Optimization)")
    print("="*60)
    
    best_robust_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train with adversarial examples
        train_loss, train_acc = train_robust(
            model, train_loader, optimizer, criterion, train_attack, device, epoch, args.adv_ratio
        )
        
        # Evaluate on clean data
        clean_acc, clean_loss = evaluate_clean(model, test_loader, device)
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss (adv): {train_loss:.4f}, Train Acc (adv): {train_acc:.2f}%")
        print(f"  Test Loss (clean): {clean_loss:.4f}, Test Acc (clean): {clean_acc:.2f}%")
        
        # Periodic robust evaluation
        if epoch % args.eval_freq == 0:
            print(f"  Running robust evaluation...")
            eval_attack = PGDAttack(
                model=model,
                epsilon=args.eval_epsilon,
                alpha=args.eval_epsilon/4,
                num_iter=10,
                device=device
            )
            robust_acc, robust_loss = evaluate_adversarial(model, test_loader, eval_attack, device)
            print(f"  Test Loss (PGD-10): {robust_loss:.4f}, Test Acc (PGD-10): {robust_acc:.2f}%")
            
            # Save best robust model
            if robust_acc > best_robust_acc:
                best_robust_acc = robust_acc
                save_checkpoint(
                    model, optimizer, epoch, robust_acc,
                    os.path.join(args.output_dir, 'models', 'robust_best.pth')
                )
        
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, clean_acc,
                os.path.join(args.output_dir, 'models', f'robust_epoch_{epoch}.pth')
            )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    # Clean accuracy
    clean_acc, _ = evaluate_clean(model, test_loader, device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # Adversarial accuracy (PGD-10)
    print("\nEvaluating against PGD-10 attack...")
    final_attack = PGDAttack(
        model=model,
        epsilon=args.eval_epsilon,
        alpha=args.eval_epsilon/4,
        num_iter=10,
        device=device
    )
    robust_acc, _ = evaluate_adversarial(model, test_loader, final_attack, device)
    print(f"Robust Accuracy (PGD-10): {robust_acc:.2f}%")
    
    # Adversarial accuracy (PGD-20)
    print("\nEvaluating against PGD-20 attack (stronger)...")
    strong_attack = PGDAttack(
        model=model,
        epsilon=args.eval_epsilon,
        alpha=args.eval_epsilon/4,
        num_iter=20,
        device=device
    )
    robust_acc_20, _ = evaluate_adversarial(model, test_loader, strong_attack, device)
    print(f"Robust Accuracy (PGD-20): {robust_acc_20:.2f}%")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, robust_acc,
        os.path.join(args.output_dir, 'models', 'robust_final.pth')
    )
    
    print("\nTraining completed!")
    print(f"Best Robust Accuracy (PGD-10): {best_robust_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust Training for CIFAR-10')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for CIFAR-10 dataset')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory for saving outputs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate (use 0.01 for pre-training, 0.1 from scratch)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Learning rate schedule
    parser.add_argument('--lr-milestones', type=int, nargs='+', default=[75, 90],
                        help='Epochs to decrease learning rate')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Learning rate decay factor')
    
    # Adversarial training parameters (inner maximization)
    parser.add_argument('--train-epsilon', type=float, default=4/255,
                        help='Perturbation bound for training (L-inf norm)')
    parser.add_argument('--train-alpha', type=float, default=2/255,
                        help='Step size for PGD during training')
    parser.add_argument('--train-steps', type=int, default=10,
                        help='Number of PGD steps during training')
    parser.add_argument('--adv-ratio', type=float, default=0.5,
                        help='Ratio of adversarial examples in mixed training (0.5 = 50-50, 0.75 = 75-25, 1.0 = pure adversarial)')
    
    # Evaluation parameters
    parser.add_argument('--eval-epsilon', type=float, default=4/255,
                        help='Epsilon for adversarial evaluation')
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluate robust accuracy every N epochs')
    
    # Pre-training parameters
    parser.add_argument('--pretrain-path', type=str, default=None,
                        help='Path to pre-trained standard model (e.g., output/models/standard_final.pth)')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--save-freq', type=int, default=25,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    main(args)
