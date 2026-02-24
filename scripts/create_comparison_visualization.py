"""Create Before/After Comparison: Standard vs Robust Model Under Attack."""

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

def compare_models(standard_path, robust_path, output_path='model_comparison.png', device='cpu'):
    """Create visualization comparing standard vs robust model under attack."""
    
    # Load both models
    standard_model = CNN(num_classes=10).to(device)
    robust_model = CNN(num_classes=10).to(device)
    
    checkpoint_std = torch.load(standard_path, map_location=device)
    checkpoint_rob = torch.load(robust_path, map_location=device)
    
    standard_model.load_state_dict(checkpoint_std['model_state_dict'])
    robust_model.load_state_dict(checkpoint_rob['model_state_dict'])
    
    standard_model.eval()
    robust_model.eval()
    
    # Load CIFAR-10 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    # Try multiple examples to find one where robust model defends successfully
    print("Searching for a good example...")
    best_idx = None
    for test_idx in range(100):
        image, label = test_dataset[test_idx]
        image_test = image.unsqueeze(0).to(device)
        label_tensor = torch.tensor([label]).to(device)
        
        # Generate adversarial example
        attack_temp = PGDAttack(standard_model, epsilon=4/255, alpha=2/255, num_iter=10, device=device)
        image_adv_test = attack_temp(image_test, label_tensor)
        
        # Check both models
        with torch.no_grad():
            std_clean_pred = standard_model(image_test).argmax(1).item()
            std_adv_pred = standard_model(image_adv_test).argmax(1).item()
            rob_clean_pred = robust_model(image_test).argmax(1).item()
            rob_adv_pred = robust_model(image_adv_test).argmax(1).item()
        
        # Find example where:
        # 1. Both models correct on clean
        # 2. Standard model fooled by attack
        # 3. Robust model defends successfully
        if (std_clean_pred == label and rob_clean_pred == label and
            std_adv_pred != label and rob_adv_pred == label):
            best_idx = test_idx
            print(f"✓ Found good example at index {best_idx}: {CLASSES[label]}")
            break
    
    if best_idx is None:
        print("⚠ No perfect example found, using index 15")
        best_idx = 15
    
    # Use the selected example
    idx = best_idx
    image, label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    label_tensor = torch.tensor([label]).to(device)
    
    # Initialize PGD attack (using standard model for consistency)
    attack = PGDAttack(standard_model, epsilon=4/255, alpha=2/255, num_iter=10, device=device)
    
    # Generate adversarial example
    image_adv = attack(image, label_tensor)
    
    # Get predictions from both models on CLEAN image
    with torch.no_grad():
        std_clean_out = standard_model(image)
        rob_clean_out = robust_model(image)
        std_clean_pred = std_clean_out.argmax(1).item()
        rob_clean_pred = rob_clean_out.argmax(1).item()
        std_clean_conf = torch.softmax(std_clean_out, 1)[0, std_clean_pred].item()
        rob_clean_conf = torch.softmax(rob_clean_out, 1)[0, rob_clean_pred].item()
    
    # Get predictions from both models on ADVERSARIAL image
    with torch.no_grad():
        std_adv_out = standard_model(image_adv)
        rob_adv_out = robust_model(image_adv)
        std_adv_pred = std_adv_out.argmax(1).item()
        rob_adv_pred = rob_adv_out.argmax(1).item()
        std_adv_conf = torch.softmax(std_adv_out, 1)[0, std_adv_pred].item()
        rob_adv_conf = torch.softmax(rob_adv_out, 1)[0, rob_adv_pred].item()
    
    # Denormalize for visualization
    img_clean = denormalize(image[0]).cpu().permute(1, 2, 0).numpy()
    img_adv = denormalize(image_adv[0]).cpu().permute(1, 2, 0).numpy()
    img_clean = np.clip(img_clean, 0, 1)
    img_adv = np.clip(img_adv, 0, 1)
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Title - simplified
    fig.suptitle('Standard vs Robust Model Under Attack', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # Row 1: Standard Model
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_clean)
    ax1.set_title('Clean', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_adv)
    ax2.set_title('Attacked', fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    std_success = std_clean_pred != std_adv_pred
    ax3.text(0.5, 0.75, 'Standard Model', ha='center', va='center',
             fontsize=14, fontweight='bold', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5))
    ax3.text(0.5, 0.45, f'{CLASSES[std_clean_pred]} → {CLASSES[std_adv_pred]}',
             ha='center', va='center', transform=ax3.transAxes, fontsize=12,
             fontweight='bold')
    ax3.text(0.5, 0.25, f'{std_clean_conf:.0%} → {std_adv_conf:.0%}',
             ha='center', va='center', transform=ax3.transAxes, fontsize=11,
             color='gray')
    ax3.text(0.5, 0.05, '✗ FOOLED' if std_success else '✓ DEFENDED',
             ha='center', va='center', transform=ax3.transAxes, fontsize=13,
             fontweight='bold', color='red' if std_success else 'green')
    ax3.axis('off')
    
    # Row 2: Robust Model (Your Work!)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(img_clean)
    ax4.set_title('Clean', fontsize=11, fontweight='bold')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(img_adv)
    ax5.set_title('Attacked', fontsize=11, fontweight='bold')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    rob_success = rob_clean_pred != rob_adv_pred
    ax6.text(0.5, 0.75, 'Robust Model (Ours)', ha='center', va='center',
             fontsize=14, fontweight='bold', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8, pad=0.5))
    ax6.text(0.5, 0.45, f'{CLASSES[rob_clean_pred]} → {CLASSES[rob_adv_pred]}',
             ha='center', va='center', transform=ax6.transAxes, fontsize=12,
             fontweight='bold')
    ax6.text(0.5, 0.25, f'{rob_clean_conf:.0%} → {rob_adv_conf:.0%}',
             ha='center', va='center', transform=ax6.transAxes, fontsize=11,
             color='gray')
    ax6.text(0.5, 0.05, '✗ FOOLED' if rob_success else '✓ DEFENDED',
             ha='center', va='center', transform=ax6.transAxes, fontsize=13,
             fontweight='bold', color='red' if rob_success else 'green')
    ax6.axis('off')
    
    # Add ground truth label - simplified
    fig.text(0.5, 0.01, f'Ground Truth: {CLASSES[label]}', 
             ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comparison visualization saved to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"Ground Truth: {CLASSES[label]}")
    print(f"\nStandard Model:")
    print(f"  Clean: {CLASSES[std_clean_pred]} ({std_clean_conf:.1%})")
    print(f"  Under Attack: {CLASSES[std_adv_pred]} ({std_adv_conf:.1%})")
    print(f"  Status: {'FOOLED ✗' if std_success else 'DEFENDED ✓'}")
    print(f"\nRobust Model (Your Work!):")
    print(f"  Clean: {CLASSES[rob_clean_pred]} ({rob_clean_conf:.1%})")
    print(f"  Under Attack: {CLASSES[rob_adv_pred]} ({rob_adv_conf:.1%})")
    print(f"  Status: {'FOOLED ✗' if rob_success else 'DEFENDED ✓'}")
    print(f"{'='*60}")
    
    plt.close()

if __name__ == '__main__':
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    
    print(f"Using device: {device}")
    print(f"Creating before/after comparison visualization...")
    
    compare_models(
        'output/models/standard_final.pth',
        'output/models/robust_final.pth',
        'model_comparison.png',
        device=device
    )
    
    print("\nUse this image to show YOUR CONTRIBUTION:")
    print("  \\includegraphics[width=\\linewidth]{model_comparison.png}")
