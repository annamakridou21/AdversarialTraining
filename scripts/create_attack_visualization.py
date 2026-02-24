"""Create PGD Attack Visualization for Presentation."""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import matplotlib.patches as mpatches

from model_arch import CNN
from attacks import PGDAttack

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def denormalize(tensor):
    """Denormalize CIFAR-10 images."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def visualize_attack(model_path, output_path='attack_visualization.png', device='cpu'):
    """Create visualization of PGD attack process."""
    
    # Load model
    model = CNN(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load CIFAR-10 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    # Select 1 example (manually picked index that works well)
    indices = [15]  # You can adjust this
    
    # Initialize PGD attack
    attack = PGDAttack(model, epsilon=4/255, alpha=2/255, num_iter=10, device=device)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
    fig.suptitle('PGD Attack Visualization (ε=4/255, 10 iterations)', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    for idx, img_idx in enumerate(indices):
        image, label = test_dataset[img_idx]
        image = image.unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = model(image)
            orig_pred = orig_output.argmax(1).item()
            orig_conf = torch.softmax(orig_output, 1)[0, orig_pred].item()
        
        # Generate adversarial example
        image_adv = attack(image, label_tensor)
        
        # Get adversarial prediction
        with torch.no_grad():
            adv_output = model(image_adv)
            adv_pred = adv_output.argmax(1).item()
            adv_conf = torch.softmax(adv_output, 1)[0, adv_pred].item()
        
        # Compute perturbation
        perturbation = image_adv - image
        
        # Denormalize for visualization
        img_clean = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
        img_adv = denormalize(image_adv[0]).cpu().permute(1, 2, 0).numpy()
        img_pert = perturbation[0].cpu().permute(1, 2, 0).numpy()
        
        # Clip to valid range
        img_clean = np.clip(img_clean, 0, 1)
        img_adv = np.clip(img_adv, 0, 1)
        
        # Enhance perturbation visibility
        img_pert_vis = (img_pert - img_pert.min()) / (img_pert.max() - img_pert.min())
        
        # Plot original image
        axes[idx, 0].imshow(img_clean)
        axes[idx, 0].set_title(f'Original\n"{CLASSES[label]}"', fontsize=10, fontweight='bold')
        axes[idx, 0].text(0.5, -0.15, f'Pred: {CLASSES[orig_pred]}\nConf: {orig_conf:.1%}',
                         ha='center', va='top', transform=axes[idx, 0].transAxes,
                         fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        axes[idx, 0].axis('off')
        
        # Plot perturbation
        axes[idx, 1].imshow(img_pert_vis, cmap='RdBu_r')
        axes[idx, 1].set_title('Perturbation\n(amplified)', fontsize=10, fontweight='bold')
        axes[idx, 1].text(0.5, -0.15, f'||δ||∞ = {perturbation.abs().max():.4f}',
                         ha='center', va='top', transform=axes[idx, 1].transAxes,
                         fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        axes[idx, 1].axis('off')
        
        # Plot adversarial image
        axes[idx, 2].imshow(img_adv)
        axes[idx, 2].set_title('Adversarial\n(barely visible)', fontsize=10, fontweight='bold')
        axes[idx, 2].axis('off')
        
        # Plot prediction
        success = orig_pred != adv_pred
        color = 'lightcoral' if success else 'lightblue'
        status = '✓ Attack Success' if success else '✗ Attack Failed'
        
        axes[idx, 3].text(0.5, 0.7, status, ha='center', va='center',
                         fontsize=12, fontweight='bold',
                         transform=axes[idx, 3].transAxes,
                         color='red' if success else 'blue')
        axes[idx, 3].text(0.5, 0.4, f'Prediction:\n{CLASSES[adv_pred]}',
                         ha='center', va='center', transform=axes[idx, 3].transAxes,
                         fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        axes[idx, 3].text(0.5, 0.15, f'Confidence:\n{adv_conf:.1%}',
                         ha='center', va='center', transform=axes[idx, 3].transAxes,
                         fontsize=9)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    import sys
    
    # Use standard model to show vulnerability
    model_path = 'output/models/standard_final.pth'
    
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    
    print(f"Using device: {device}")
    print(f"Creating attack visualization using {model_path}...")
    
    visualize_attack(model_path, 'attack_visualization.png', device=device)
    
    print("\nYou can now include this image in your presentation:")
    print("  \\includegraphics[width=\\linewidth]{attack_visualization.png}")
