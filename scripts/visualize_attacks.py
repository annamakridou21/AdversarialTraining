"""Visualize Adversarial Examples.

Creates side-by-side comparisons of clean vs adversarial images.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from model_arch import CNN
from utils import get_cifar10_loaders
from attacks import PGDAttack


# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """Denormalize image tensor for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_adversarial_examples(model, test_loader, device, num_images=10, epsilon=4/255):
    """Generate and visualize adversarial examples.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to run on
        num_images: Number of example pairs to visualize
        epsilon: Attack strength
    """
    # Create attack
    attack = PGDAttack(
        model=model,
        epsilon=epsilon,
        alpha=epsilon/4,
        num_iter=10,
        device=device
    )
    
    model.eval()
    
    # Collect examples
    images_list = []
    images_adv_list = []
    labels_list = []
    pred_clean_list = []
    pred_adv_list = []
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Get clean predictions
        with torch.no_grad():
            outputs_clean = model(images)
            _, pred_clean = outputs_clean.max(1)
        
        # Generate adversarial examples
        images_adv = attack(images, labels, random_start=True)
        
        # Get adversarial predictions
        with torch.no_grad():
            outputs_adv = model(images_adv)
            _, pred_adv = outputs_adv.max(1)
        
        # Store examples where attack was successful (clean correct, adv wrong)
        successful = pred_clean.eq(labels) & ~pred_adv.eq(labels)
        
        if successful.sum() > 0:
            images_list.append(images[successful].cpu())
            images_adv_list.append(images_adv[successful].cpu())
            labels_list.append(labels[successful].cpu())
            pred_clean_list.append(pred_clean[successful].cpu())
            pred_adv_list.append(pred_adv[successful].cpu())
        
        if sum(len(x) for x in images_list) >= num_images:
            break
    
    # Concatenate collected examples
    images_all = torch.cat(images_list, dim=0)[:num_images]
    images_adv_all = torch.cat(images_adv_list, dim=0)[:num_images]
    labels_all = torch.cat(labels_list, dim=0)[:num_images]
    pred_clean_all = torch.cat(pred_clean_list, dim=0)[:num_images]
    pred_adv_all = torch.cat(pred_adv_list, dim=0)[:num_images]
    
    # Calculate perturbation
    perturbation = (images_adv_all - images_all).abs()
    
    # Create visualization
    n = min(num_images, len(images_all))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Denormalize images
        img_clean = denormalize(images_all[i]).permute(1, 2, 0).numpy()
        img_adv = denormalize(images_adv_all[i]).permute(1, 2, 0).numpy()
        img_pert = perturbation[i].permute(1, 2, 0).numpy()
        img_pert = img_pert / img_pert.max()  # Normalize for visibility
        
        # Clean image
        axes[i, 0].imshow(img_clean)
        axes[i, 0].set_title(f'Clean\nTrue: {CIFAR10_CLASSES[labels_all[i]]}\n'
                            f'Pred: {CIFAR10_CLASSES[pred_clean_all[i]]}',
                            fontsize=10, color='green')
        axes[i, 0].axis('off')
        
        # Perturbation (amplified for visibility)
        axes[i, 1].imshow(img_pert)
        axes[i, 1].set_title(f'Perturbation\n(amplified)\nε={epsilon*255:.1f}/255',
                            fontsize=10)
        axes[i, 1].axis('off')
        
        # Adversarial image
        axes[i, 2].imshow(img_adv)
        axes[i, 2].set_title(f'Adversarial\nTrue: {CIFAR10_CLASSES[labels_all[i]]}\n'
                            f'Pred: {CIFAR10_CLASSES[pred_adv_all[i]]}',
                            fontsize=10, color='red')
        axes[i, 2].axis('off')
        
        # Difference (clean - adversarial)
        diff = np.abs(img_clean - img_adv)
        diff = diff / diff.max()  # Normalize
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title(f'Absolute Diff\n(amplified)',
                            fontsize=10)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    return fig


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    
    # Initialize model
    model = CNN().to(device)
    
    # Load checkpoint
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"\nGenerating {args.num_images} adversarial examples...")
    print(f"Attack: PGD-10 with ε = {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
    
    # Generate visualization
    fig = visualize_adversarial_examples(
        model, test_loader, device,
        num_images=args.num_images,
        epsilon=args.epsilon
    )
    
    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    if args.show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize adversarial examples')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-images', type=int, default=5,
                        help='Number of example pairs to visualize')
    parser.add_argument('--epsilon', type=float, default=4/255,
                        help='Attack strength (epsilon)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='./output/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--output-name', type=str, default='adversarial_examples.png',
                        help='Output filename')
    parser.add_argument('--show', action='store_true',
                        help='Display the plot')
    
    args = parser.parse_args()
    main(args)
