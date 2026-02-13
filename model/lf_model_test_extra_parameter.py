"""
LF-PINN with configurable input features for theta_net.
Allows testing which inputs improve theta prediction.
"""

import torch
import torch.nn as nn
import numpy as np


class LFPinn_InputFeatures_Test(nn.Module):
    """
    LF-PINN with configurable theta_net inputs.
    
    Available features:
        'x'         - spatial coordinate
        't'         - current time
        't_next'    - next time step
        'h'         - step size (t_next - t)
        't_norm'    - normalized time t / t_max
        'grad_norm' - |∂y/∂x| (gradient magnitude)
        'y'         - current solution value
        'y_abs'     - |y| (absolute value)
        'laplacian' - ∂²y/∂x² (curvature, expensive!)
    """
    
    AVAILABLE_FEATURES = ['x', 't', 't_next', 'h', 't_norm', 'grad_norm', 'y', 'y_abs', 'laplacian']
    
    def __init__(self, pde_type: str, 
                 input_features: list = None,
                 n_steps: int = 10, 
                 theta_hidden_dim: int = 2,
                 n_iterations: int = 2, 
                 lr: float = 0.001, 
                 initial_theta: float = 0.5,
                 t_max: float = 1.0):  # for t_norm
        super().__init__()
        
        self.pde_type = pde_type
        self.n_steps = n_steps
        self.is_wave = (pde_type == 'wave')
        self.n_iterations = n_iterations
        self.lr = lr
        self.initial_theta = initial_theta
        self.t_max = t_max
        
        # Default features (current implementation)
        if input_features is None:
            input_features = ['x', 't', 't_next', 'grad_norm']
        
        # Validate features
        for f in input_features:
            if f not in self.AVAILABLE_FEATURES:
                raise ValueError(f"Unknown feature: {f}. Available: {self.AVAILABLE_FEATURES}")
        
        self.input_features = input_features
        self.n_inputs = len(input_features)
        
        # Theta network with dynamic input size
        self.theta_net = nn.Sequential(
            nn.Linear(self.n_inputs, theta_hidden_dim), nn.Tanh(),
            nn.Linear(theta_hidden_dim, 1), nn.Sigmoid()
        )
        
        self._init_params()
        self._init_theta_bias(initial_theta)
        self._print_info()
    
    def _init_theta_bias(self, target_theta: float):
        """Initialize bias so that initial theta_net output ≈ target_theta"""
        with torch.no_grad():
            target_clamped = np.clip(target_theta, 0.01, 0.99)
            bias_value = np.log(target_clamped / (1 - target_clamped))
            nn.init.xavier_normal_(self.theta_net[-2].weight, gain=0.1)
            self.theta_net[-2].bias.fill_(bias_value)
    
    def _init_params(self):
        """Initialize PDE parameters"""
        if self.pde_type == 'heat':
            self.register_buffer('alpha', torch.tensor(1.0))
        elif self.pde_type == 'wave':
            self.register_buffer('c', torch.tensor(1.0))
        elif self.pde_type == 'burgers':
            self.register_buffer('nu', torch.tensor(0.01))
        elif self.pde_type == 'reaction_diffusion':
            self.register_buffer('D', torch.tensor(0.01))
            self.register_buffer('r', torch.tensor(1.0))
    
    def _print_info(self):
        n_params = sum(p.numel() for p in self.theta_net.parameters())
        print(f"\n{'='*60}")
        print(f"LF-PINN Input Features Test")
        print(f"PDE: {self.pde_type} | Steps: {self.n_steps} | Theta params: {n_params}")
        print(f"Input features ({self.n_inputs}): {self.input_features}")
        print(f"Initial theta: {self.initial_theta}")
        print(f"{'='*60}\n")
    
    def initial_condition(self, x):
        if self.pde_type == 'heat':
            return torch.sin(np.pi * x)
        elif self.pde_type == 'wave':
            return torch.cat([torch.sin(np.pi * x), torch.zeros_like(x)], dim=1)
        elif self.pde_type == 'burgers':
            return -torch.sin(np.pi * x)
        elif self.pde_type == 'reaction_diffusion':
            return 0.5 * (1 + torch.tanh(x / np.sqrt(2 * 0.01)))
        return torch.zeros_like(x)
    
    def compute_rhs(self, x, t, state):
        if not state.requires_grad:
            return torch.zeros_like(state)
        
        if self.pde_type == 'wave':
            u, v = state[:, 0:1], state[:, 1:2]
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x)
            u_xx = u_xx if u_xx is not None else torch.zeros_like(x)
            return torch.cat([v, self.c**2 * u_xx], dim=1)
        
        u = state
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x)
        
        u_x = u_x if u_x is not None else torch.zeros_like(x)
        u_xx = u_xx if u_xx is not None else torch.zeros_like(x)
        
        if self.pde_type == 'heat':
            return self.alpha * u_xx
        elif self.pde_type == 'burgers':
            return -u * u_x + self.nu * u_xx
        elif self.pde_type == 'reaction_diffusion':
            return self.D * u_xx + self.r * u * (1 - u)
    
    def _compute_theta_inputs(self, x, t, t_next, h, y, y_x, y_xx):
        """
        Compute input tensor for theta_net based on selected features.
        All computations are detached for theta_net input.
        """
        features = []
        
        y_scalar = y[:, 0:1] if self.is_wave else y
        
        for feat in self.input_features:
            if feat == 'x':
                features.append(x.detach())
            
            elif feat == 't':
                features.append(t.detach())
            
            elif feat == 't_next':
                features.append(t_next.detach())
            
            elif feat == 'h':
                features.append(h.detach())
            
            elif feat == 't_norm':
                # Normalized time in [0, 1]
                features.append((t / self.t_max).detach())
            
            elif feat == 'grad_norm':
                # |∂y/∂x|, normalized with tanh
                if y_x is not None:
                    grad_norm = torch.tanh(torch.abs(y_x) / 5.0)
                else:
                    grad_norm = torch.zeros_like(x)
                features.append(grad_norm.detach())
            
            elif feat == 'y':
                # Current solution value
                features.append(y_scalar.detach())
            
            elif feat == 'y_abs':
                # |y|, normalized
                features.append(torch.tanh(torch.abs(y_scalar)).detach())
            
            elif feat == 'laplacian':
                # ∂²y/∂x², normalized
                if y_xx is not None:
                    laplacian = torch.tanh(y_xx / 10.0)
                else:
                    laplacian = torch.zeros_like(x)
                features.append(laplacian.detach())
        
        return torch.cat(features, dim=1)
    
    def forward(self, x, t_end):
        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype
        
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor(t_end, dtype=dtype, device=device)
        if t_end.dim() == 0:
            t_end = t_end.expand(batch_size, 1)
        elif t_end.shape[0] == 1 and batch_size > 1:
            t_end = t_end.expand(batch_size, 1)
        
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        
        h = t_end / self.n_steps
        
        y = self.initial_condition(x)
        t = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        
        for _ in range(self.n_steps):
            t_next = t + h
            
            y_scalar = y[:, 0:1] if self.is_wave else y
            
            # Compute gradients if needed
            y_x = None
            y_xx = None
            
            if any(f in self.input_features for f in ['grad_norm', 'laplacian']):
                y_x = torch.autograd.grad(
                    y_scalar.sum(), x,
                    create_graph=True, retain_graph=True, allow_unused=True
                )[0]
                if y_x is None:
                    y_x = torch.zeros_like(x)
                
                if 'laplacian' in self.input_features:
                    y_xx = torch.autograd.grad(
                        y_x.sum(), x,
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    if y_xx is None:
                        y_xx = torch.zeros_like(x)
            
            # Build theta input
            theta_input = self._compute_theta_inputs(x, t, t_next, h, y, y_x, y_xx)
            theta = self.theta_net(theta_input)
            
            # Fixed-point iteration
            f_curr = self.compute_rhs(x, t, y)
            
            y_new = y.clone()
            for _ in range(self.n_iterations):
                f_next = self.compute_rhs(x, t_next, y_new)
                y_new = y + h * ((1 - theta) * f_curr + theta * f_next)
            
            y = y_new
            t = t_next
        
        return y[:, 0:1] if self.is_wave else y
    
    def pde_loss(self, x_col, t_col):
        x_col = x_col.detach().clone().requires_grad_(True)
        t_col = t_col.detach().clone().requires_grad_(True)
        u = self.forward(x_col, t_col)
        
        u_x = torch.autograd.grad(u.sum(), x_col, create_graph=True, allow_unused=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x_col, create_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x_col)
        u_t = torch.autograd.grad(u.sum(), t_col, create_graph=True, allow_unused=True)[0]
        
        u_x = u_x if u_x is not None else torch.zeros_like(x_col)
        u_xx = u_xx if u_xx is not None else torch.zeros_like(x_col)
        u_t = u_t if u_t is not None else torch.zeros_like(t_col)
        
        if self.pde_type == 'heat':
            residual = u_t - self.alpha * u_xx
        elif self.pde_type == 'burgers':
            residual = u_t + u * u_x - self.nu * u_xx
        elif self.pde_type == 'reaction_diffusion':
            residual = u_t - self.D * u_xx - self.r * u * (1 - u)
        elif self.pde_type == 'wave':
            u_tt = torch.autograd.grad(u_t.sum(), t_col, create_graph=True, allow_unused=True)[0]
            residual = (u_tt if u_tt is not None else torch.zeros_like(t_col)) - self.c**2 * u_xx
        
        return torch.mean(residual**2)
    
    def boundary_loss(self, domain, n_bc=10):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        t_bc = torch.rand(n_bc, 1, device=device) * (t_max - t_min) + t_min
        x_left = torch.full((n_bc, 1), x_min, device=device)
        x_right = torch.full((n_bc, 1), x_max, device=device)
        
        u_left = self.forward(x_left, t_bc)
        u_right = self.forward(x_right, t_bc)
        
        if self.pde_type in ['heat', 'wave']:
            return torch.mean(u_left**2) + torch.mean(u_right**2)
        elif self.pde_type == 'reaction_diffusion':
            return torch.mean(u_left**2) + torch.mean((u_right - 1.0)**2)
        elif self.pde_type == 'burgers':
            return torch.mean((u_left - u_right)**2)
        return torch.tensor(0.0, device=device)
    
    def initial_condition_loss(self, x_ic, n_ic=10):
        if x_ic is None:
            x_ic = torch.rand(n_ic, 1, device=next(self.parameters()).device)
        
        t_zero = torch.zeros_like(x_ic)
        u_pred = self.forward(x_ic, t_zero)
        u_true = self.initial_condition(x_ic)
        
        if self.is_wave:
            u_true = u_true[:, 0:1]
        
        return torch.mean((u_pred - u_true)**2)
    
    def total_loss(self, domain, n_collocation=30,
                   lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
        t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
        
        loss_pde = self.pde_loss(x_col, t_col)
        loss_bc = self.boundary_loss(domain, n_bc=n_collocation)
        loss_ic = self.initial_condition_loss(x_col, n_ic=n_collocation)
        
        total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        
        return total, {
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
            'total': total.item()
        }
    
    def get_params(self):
        return {
            'n_steps': self.n_steps,
            'n_iterations': self.n_iterations,
            'lr': self.lr,
            'input_features': self.input_features,
            'n_inputs': self.n_inputs,
            'initial_theta': self.initial_theta
        }
    
    def get_theta_statistics(self, domain, n_samples=100):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_sample = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
        t_sample = torch.rand(n_samples, 1, device=device) * (t_max - t_min) + t_min
        h = torch.full((n_samples, 1), (t_max - t_min) / self.n_steps, device=device)
        t_next = torch.clamp(t_sample + h, max=t_max)
        
        with torch.no_grad():
            y_approx = self.initial_condition(x_sample)
            if self.is_wave:
                y_approx = y_approx[:, 0:1]
            
            # Dummy values for gradients
            y_x = torch.zeros_like(x_sample)
            y_xx = torch.zeros_like(x_sample)
            
            theta_input = self._compute_theta_inputs(x_sample, t_sample, t_next, h, y_approx, y_x, y_xx)
            theta_values = self.theta_net(theta_input)
        
        return {
            'mean': theta_values.mean().item(),
            'std': theta_values.std().item(),
            'min': theta_values.min().item(),
            'max': theta_values.max().item()
        }