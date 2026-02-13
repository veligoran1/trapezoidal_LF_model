import torch
import torch.nn as nn
import numpy as np


class LFPinn_LossWeighting_Test(nn.Module):
    """
    LF-PINN with configurable loss weighting strategies.
    
    Available strategies:
        'fixed'         - Fixed weights (baseline): λ_pde, λ_bc, λ_ic
        'gradual'       - Gradual increase of BC/IC weights over epochs
        'inverse'       - Inverse weighting: λ_i = 1 / (L_i + ε)
        'softadapt'     - SoftAdapt: weights based on loss change rates
        'relobralo'     - ReLoBRaLo: relative loss balancing with random lookback
        'ntk'           - NTK-inspired: approximate gradient magnitude balancing
        'self_adaptive' - Learnable weights as parameters
        'causal'        - Causal weighting: higher weight for early times
    """
    
    AVAILABLE_STRATEGIES = [
        'fixed', 'gradual', 'inverse', 'softadapt', 
        'relobralo', 'ntk', 'self_adaptive', 'causal'
    ]
    
    def __init__(self, pde_type: str,
                 weighting_strategy: str = 'fixed',
                 # Fixed weights (used as base/initial for adaptive methods too)
                 lambda_pde: float = 1.0,
                 lambda_bc: float = 10.0,
                 lambda_ic: float = 10.0,
                 # Strategy-specific params
                 gradual_epochs: int = 100,      # for 'gradual'
                 softadapt_beta: float = 0.9,    # for 'softadapt'
                 relobralo_alpha: float = 0.9,   # for 'relobralo'
                 relobralo_tau: float = 1.0,     # temperature for 'relobralo'
                 causal_epsilon: float = 1.0,    # for 'causal'
                 # Model params
                 n_steps: int = 10,
                 theta_hidden_dim: int = 2,
                 n_iterations: int = 2,
                 lr: float = 0.001,
                 initial_theta: float = 0.5):
        super().__init__()
        
        self.pde_type = pde_type
        self.n_steps = n_steps
        self.is_wave = (pde_type == 'wave')
        self.n_iterations = n_iterations
        self.lr = lr
        self.initial_theta = initial_theta
        
        # Weighting strategy
        if weighting_strategy not in self.AVAILABLE_STRATEGIES:
            raise ValueError(f"Unknown strategy: {weighting_strategy}. "
                           f"Available: {self.AVAILABLE_STRATEGIES}")
        self.weighting_strategy = weighting_strategy
        
        # Base weights
        self.lambda_pde_base = lambda_pde
        self.lambda_bc_base = lambda_bc
        self.lambda_ic_base = lambda_ic
        
        # Strategy-specific params
        self.gradual_epochs = gradual_epochs
        self.softadapt_beta = softadapt_beta
        self.relobralo_alpha = relobralo_alpha
        self.relobralo_tau = relobralo_tau
        self.causal_epsilon = causal_epsilon
        
        # For adaptive methods: track loss history
        self.loss_history = {'pde': [], 'bc': [], 'ic': []}
        self.current_epoch = 0
        
        # For self-adaptive: learnable log-weights
        if weighting_strategy == 'self_adaptive':
            self.log_lambda_pde = nn.Parameter(torch.tensor(np.log(lambda_pde)))
            self.log_lambda_bc = nn.Parameter(torch.tensor(np.log(lambda_bc)))
            self.log_lambda_ic = nn.Parameter(torch.tensor(np.log(lambda_ic)))
        
        # Theta network
        self.theta_net = nn.Sequential(
            nn.Linear(4, theta_hidden_dim), nn.Tanh(),
            nn.Linear(theta_hidden_dim, 1), nn.Sigmoid()
        )
        
        self._init_params()
        self._init_theta_bias(initial_theta)
        self._print_info()
    
    def _init_theta_bias(self, target_theta: float):
        with torch.no_grad():
            target_clamped = np.clip(target_theta, 0.01, 0.99)
            bias_value = np.log(target_clamped / (1 - target_clamped))
            nn.init.xavier_normal_(self.theta_net[-2].weight, gain=0.1)
            self.theta_net[-2].bias.fill_(bias_value)
    
    def _init_params(self):
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
        extra = ""
        if self.weighting_strategy == 'self_adaptive':
            extra = " (+3 learnable weights)"
        print(f"\n{'='*60}")
        print(f"LF-PINN Loss Weighting Test")
        print(f"PDE: {self.pde_type} | Strategy: {self.weighting_strategy}")
        print(f"Base weights: λ_pde={self.lambda_pde_base}, λ_bc={self.lambda_bc_base}, λ_ic={self.lambda_ic_base}")
        print(f"Theta params: {n_params}{extra}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # WEIGHT COMPUTATION
    # ============================================================
    
    def get_weights(self, loss_pde: torch.Tensor, loss_bc: torch.Tensor, 
                    loss_ic: torch.Tensor) -> tuple:
        """
        Compute weights based on current strategy.
        Returns (lambda_pde, lambda_bc, lambda_ic)
        """
        
        if self.weighting_strategy == 'fixed':
            return self._weights_fixed()
        
        elif self.weighting_strategy == 'gradual':
            return self._weights_gradual()
        
        elif self.weighting_strategy == 'inverse':
            return self._weights_inverse(loss_pde, loss_bc, loss_ic)
        
        elif self.weighting_strategy == 'softadapt':
            return self._weights_softadapt(loss_pde, loss_bc, loss_ic)
        
        elif self.weighting_strategy == 'relobralo':
            return self._weights_relobralo(loss_pde, loss_bc, loss_ic)
        
        elif self.weighting_strategy == 'ntk':
            return self._weights_ntk(loss_pde, loss_bc, loss_ic)
        
        elif self.weighting_strategy == 'self_adaptive':
            return self._weights_self_adaptive()
        
        elif self.weighting_strategy == 'causal':
            # Causal is handled differently in total_loss
            return self._weights_fixed()
        
        return self._weights_fixed()
    
    def _weights_fixed(self):
        """Fixed weights - baseline"""
        return self.lambda_pde_base, self.lambda_bc_base, self.lambda_ic_base
    
    def _weights_gradual(self):
        """
        Gradual: start with equal weights, gradually increase BC/IC.
        λ_bc(t) = λ_bc_base * min(1, epoch / gradual_epochs)
        """
        progress = min(1.0, self.current_epoch / self.gradual_epochs)
        lambda_pde = self.lambda_pde_base
        lambda_bc = self.lambda_bc_base * progress + 1.0 * (1 - progress)
        lambda_ic = self.lambda_ic_base * progress + 1.0 * (1 - progress)
        return lambda_pde, lambda_bc, lambda_ic
    
    def _weights_inverse(self, loss_pde, loss_bc, loss_ic, eps=1e-8):
        """
        Inverse weighting: λ_i = 1 / (L_i + ε)
        Normalized so that sum = sum of base weights
        """
        with torch.no_grad():
            w_pde = 1.0 / (loss_pde.item() + eps)
            w_bc = 1.0 / (loss_bc.item() + eps)
            w_ic = 1.0 / (loss_ic.item() + eps)
            
            # Normalize
            total_base = self.lambda_pde_base + self.lambda_bc_base + self.lambda_ic_base
            total_w = w_pde + w_bc + w_ic
            
            lambda_pde = w_pde / total_w * total_base
            lambda_bc = w_bc / total_w * total_base
            lambda_ic = w_ic / total_w * total_base
            
        return lambda_pde, lambda_bc, lambda_ic
    
    def _weights_softadapt(self, loss_pde, loss_bc, loss_ic):
        """
        SoftAdapt: weights based on exponential moving average of loss ratios.
        From: "SoftAdapt: Techniques for Adaptive Loss Weighting" (Heydari et al.)
        """
        # Update history
        self.loss_history['pde'].append(loss_pde.item())
        self.loss_history['bc'].append(loss_bc.item())
        self.loss_history['ic'].append(loss_ic.item())
        
        if len(self.loss_history['pde']) < 2:
            return self._weights_fixed()
        
        with torch.no_grad():
            # Compute loss change rates
            rate_pde = self.loss_history['pde'][-1] / (self.loss_history['pde'][-2] + 1e-8)
            rate_bc = self.loss_history['bc'][-1] / (self.loss_history['bc'][-2] + 1e-8)
            rate_ic = self.loss_history['ic'][-1] / (self.loss_history['ic'][-2] + 1e-8)
            
            # Softmax over rates (higher rate = needs more attention)
            rates = np.array([rate_pde, rate_bc, rate_ic])
            exp_rates = np.exp(self.softadapt_beta * rates)
            weights = exp_rates / exp_rates.sum()
            
            # Scale by base weights sum
            total_base = self.lambda_pde_base + self.lambda_bc_base + self.lambda_ic_base
            lambda_pde = weights[0] * total_base
            lambda_bc = weights[1] * total_base
            lambda_ic = weights[2] * total_base
            
        return lambda_pde, lambda_bc, lambda_ic
    
    def _weights_relobralo(self, loss_pde, loss_bc, loss_ic):
        """
        ReLoBRaLo: Relative Loss Balancing with Random Lookback.
        From: "Multi-Objective Loss Balancing for Physics-Informed Deep Learning"
        """
        self.loss_history['pde'].append(loss_pde.item())
        self.loss_history['bc'].append(loss_bc.item())
        self.loss_history['ic'].append(loss_ic.item())
        
        if len(self.loss_history['pde']) < 2:
            return self._weights_fixed()
        
        with torch.no_grad():
            # Random lookback
            lookback = np.random.randint(1, min(len(self.loss_history['pde']), 10) + 1)
            
            # Compute relative losses
            rel_pde = self.loss_history['pde'][-1] / (self.loss_history['pde'][-lookback] + 1e-8)
            rel_bc = self.loss_history['bc'][-1] / (self.loss_history['bc'][-lookback] + 1e-8)
            rel_ic = self.loss_history['ic'][-1] / (self.loss_history['ic'][-lookback] + 1e-8)
            
            # Softmax with temperature
            rels = np.array([rel_pde, rel_bc, rel_ic])
            exp_rels = np.exp(rels / self.relobralo_tau)
            weights = exp_rels / exp_rels.sum()
            
            # EMA update
            alpha = self.relobralo_alpha
            if not hasattr(self, '_relobralo_weights'):
                self._relobralo_weights = np.array([1/3, 1/3, 1/3])
            
            self._relobralo_weights = alpha * self._relobralo_weights + (1 - alpha) * weights
            
            total_base = self.lambda_pde_base + self.lambda_bc_base + self.lambda_ic_base
            lambda_pde = self._relobralo_weights[0] * total_base
            lambda_bc = self._relobralo_weights[1] * total_base
            lambda_ic = self._relobralo_weights[2] * total_base
            
        return lambda_pde, lambda_bc, lambda_ic
    
    def _weights_ntk(self, loss_pde, loss_bc, loss_ic, eps=1e-8):
        """
        NTK-inspired: balance gradients approximately.
        Simplified version: weight by inverse sqrt of loss.
        Full NTK requires computing gradient norms which is expensive.
        """
        with torch.no_grad():
            # Approximate gradient magnitude ~ sqrt(loss)
            g_pde = np.sqrt(loss_pde.item() + eps)
            g_bc = np.sqrt(loss_bc.item() + eps)
            g_ic = np.sqrt(loss_ic.item() + eps)
            
            # Balance: weight inversely to gradient magnitude
            w_pde = 1.0 / g_pde
            w_bc = 1.0 / g_bc
            w_ic = 1.0 / g_ic
            
            total_base = self.lambda_pde_base + self.lambda_bc_base + self.lambda_ic_base
            total_w = w_pde + w_bc + w_ic
            
            lambda_pde = w_pde / total_w * total_base
            lambda_bc = w_bc / total_w * total_base
            lambda_ic = w_ic / total_w * total_base
            
        return lambda_pde, lambda_bc, lambda_ic
    
    def _weights_self_adaptive(self):
        """
        Self-adaptive: learnable log-weights.
        λ_i = exp(log_λ_i) ensures positivity.
        """
        lambda_pde = torch.exp(self.log_lambda_pde)
        lambda_bc = torch.exp(self.log_lambda_bc)
        lambda_ic = torch.exp(self.log_lambda_ic)
        return lambda_pde, lambda_bc, lambda_ic
    
    # ============================================================
    # FORWARD & LOSS
    # ============================================================
    
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
            y_for_theta = y[:, 0:1] if self.is_wave else y
            
            y_x = torch.autograd.grad(
                y_for_theta.sum(), x,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if y_x is None:
                y_x = torch.zeros_like(x)
            grad_norm = torch.tanh(torch.abs(y_x) / 5.0)
            
            theta = self.theta_net(torch.cat([
                x.detach(), t.detach(), t_next.detach(), grad_norm.detach()
            ], dim=1))
            
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
    
    def pde_loss_causal(self, x_col, t_col, t_max):
        """
        Causal PDE loss: weight residuals by exp(-ε * cumulative_loss).
        Points with lower t get higher weight.
        """
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
        
        # Causal weights: higher weight for smaller t
        t_normalized = t_col / t_max
        causal_weights = torch.exp(-self.causal_epsilon * t_normalized)
        causal_weights = causal_weights / causal_weights.mean()  # normalize
        
        return torch.mean(causal_weights * residual**2)
    
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
    
    def total_loss(self, domain, n_collocation=30):
        """
        Compute total loss with current weighting strategy.
        """
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
        t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
        
        # Compute raw losses
        if self.weighting_strategy == 'causal':
            loss_pde = self.pde_loss_causal(x_col, t_col, t_max)
        else:
            loss_pde = self.pde_loss(x_col, t_col)
        
        loss_bc = self.boundary_loss(domain, n_bc=n_collocation)
        loss_ic = self.initial_condition_loss(x_col, n_ic=n_collocation)
        
        # Get adaptive weights
        lambda_pde, lambda_bc, lambda_ic = self.get_weights(loss_pde, loss_bc, loss_ic)
        
        # Handle tensor vs float weights
        if isinstance(lambda_pde, torch.Tensor):
            total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
            weights_dict = {
                'lambda_pde': lambda_pde.item(),
                'lambda_bc': lambda_bc.item(),
                'lambda_ic': lambda_ic.item()
            }
        else:
            total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
            weights_dict = {
                'lambda_pde': lambda_pde,
                'lambda_bc': lambda_bc,
                'lambda_ic': lambda_ic
            }
        
        return total, {
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
            'total': total.item(),
            **weights_dict
        }
    
    def step_epoch(self):
        """Call this after each epoch to update internal state"""
        self.current_epoch += 1
    
    def reset_state(self):
        """Reset internal state for new training run"""
        self.current_epoch = 0
        self.loss_history = {'pde': [], 'bc': [], 'ic': []}
        if hasattr(self, '_relobralo_weights'):
            del self._relobralo_weights
    
    def get_params(self):
        params = {
            'n_steps': self.n_steps,
            'n_iterations': self.n_iterations,
            'lr': self.lr,
            'weighting_strategy': self.weighting_strategy,
            'initial_theta': self.initial_theta
        }
        if self.weighting_strategy == 'self_adaptive':
            params['learned_lambda_pde'] = torch.exp(self.log_lambda_pde).item()
            params['learned_lambda_bc'] = torch.exp(self.log_lambda_bc).item()
            params['learned_lambda_ic'] = torch.exp(self.log_lambda_ic).item()
        return params
    
    def get_theta_statistics(self, domain, n_samples=100):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_sample = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
        t_sample = torch.rand(n_samples, 1, device=device) * (t_max - t_min) + t_min
        h = (t_max - t_min) / self.n_steps
        t_next = torch.clamp(t_sample + h, max=t_max)
        
        with torch.no_grad():
            grad_norm = torch.zeros_like(x_sample)
            theta_values = self.theta_net(torch.cat([
                x_sample, t_sample, t_next, grad_norm
            ], dim=1))
        
        return {
            'mean': theta_values.mean().item(),
            'std': theta_values.std().item(),
            'min': theta_values.min().item(),
            'max': theta_values.max().item()
        }
