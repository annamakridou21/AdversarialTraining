"""Adversarial Attack Implementations.

Implements Projected Gradient Descent (PGD) attack from scratch
to solve the inner maximization problem in robust optimization.
"""

import torch
import torch.nn as nn


class PGDAttack:
    """Projected Gradient Descent (PGD) Attack.
    
    Solves the inner maximization problem:
        max_{δ ∈ S} L(θ, x + δ, y)
    where S = {δ : ||δ||_∞ ≤ ε}
    
    This is implemented via iterative gradient ascent with projection:
        x_{t+1} = Π_S (x_t + α · sign(∇_x L(θ, x_t, y)))
    """
    
    def __init__(self, model, epsilon=4/255, alpha=2/255, num_iter=10, device='cpu'):
        """
        Args:
            model: Neural network model to attack
            epsilon: Maximum perturbation bound (L-infinity norm)
            alpha: Step size for each iteration
            num_iter: Number of PGD iterations
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def project(self, x_adv, x_natural):
        """Project adversarial example onto feasible set.
        
        Projects onto the L-infinity ball: ||x_adv - x_natural||_∞ ≤ ε
        Also ensures x_adv stays in valid image range [0, 1]
        
        Args:
            x_adv: Adversarial examples
            x_natural: Original natural examples
            
        Returns:
            Projected adversarial examples
        """
        # Clamp perturbation to epsilon ball
        delta = torch.clamp(x_adv - x_natural, -self.epsilon, self.epsilon)
        # Add back to natural image and clamp to valid range
        x_adv = torch.clamp(x_natural + delta, 0, 1)
        return x_adv
    
    def generate(self, x, y, random_start=True):
        """Generate adversarial examples using PGD.
        
        Args:
            x: Natural input images (batch_size, channels, height, width)
            y: True labels (batch_size,)
            random_start: Whether to start from random point in epsilon ball
            
        Returns:
            Adversarial examples that maximize the loss
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Initialize adversarial example
        if random_start:
            # Start from random point in epsilon ball
            delta = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        else:
            # Start from natural example
            x_adv = x.clone()
        
        # PGD iterations
        for _ in range(self.num_iter):
            x_adv.requires_grad = True
            
            # Forward pass
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y)
            
            # Backward pass to compute gradient
            self.model.zero_grad()
            loss.backward()
            
            # Gradient ascent step (maximize loss)
            with torch.no_grad():
                # Use sign of gradient for bounded perturbation
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + self.alpha * grad_sign
                
                # Project back onto feasible set
                x_adv = self.project(x_adv, x)
        
        return x_adv.detach()
    
    def __call__(self, x, y, random_start=True):
        """Make the attack callable."""
        return self.generate(x, y, random_start)


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) Attack.
    
    A simpler one-step variant of PGD, useful for faster evaluation.
    """
    
    def __init__(self, model, epsilon=4/255, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def generate(self, x, y):
        """Generate adversarial examples using FGSM."""
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x_adv)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # One-step gradient ascent
        with torch.no_grad():
            x_adv = x_adv + self.epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()
    
    def __call__(self, x, y):
        return self.generate(x, y)
