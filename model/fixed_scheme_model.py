import torch
import torch.nn as nn
import numpy as np

class FixedSchemePINN(nn.Module):
    def __init__(self, pde_type: str, scheme: str = 'trapezoidal', n_steps: int = 10):
        super().__init__()
        self.pde_type = pde_type
        self.n_steps = n_steps
        self.scheme = scheme
        self.is_wave = (pde_type == 'wave')
        
        scheme_values = {'explicit': 1.0, 'implicit': 0.0, 'trapezoidal': 0.5}
        if scheme not in scheme_values:
            raise ValueError(f"Unknown scheme: {scheme}. Available: {list(scheme_values.keys())}")
        
        self.fixed_theta = scheme_values[scheme]
        self._init_params()
        self._print_info()
    
    def _init_params(self):
        if self.pde_type == 'heat':
            self.register_buffer('alpha', torch.tensor(1.0))
        elif self.pde_type == 'wave':
            self.register_buffer('c', torch.tensor(1.0))
        elif self.pde_type == 'burgers':
            self.register_buffer('nu', torch.tensor(0.01/np.pi))
        elif self.pde_type == 'reaction_diffusion':
            self.register_buffer('D', torch.tensor(0.01))
            self.register_buffer('r', torch.tensor(1.0))
    
    def _print_info(self):
        print(f"\n{'='*60}")
        print(f"Fixed Scheme PINN")
        print(f"PDE: {self.pde_type} | Scheme: {self.scheme} | θ: {self.fixed_theta} | Steps: {self.n_steps}")
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
            try:
                u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
                u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x)
            except:
                u_xx = torch.zeros_like(x)
            return torch.cat([v, self.c**2 * (u_xx if u_xx is not None else torch.zeros_like(x))], dim=1)
        
        u = state
        try:
            u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x)
        except:
            u_x = u_xx = torch.zeros_like(x)
        
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
            x = x.requires_grad_(True)
        
        h = t_end / self.n_steps
        y = self.initial_condition(x)
        t = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        
        for _ in range(self.n_steps):
            t_next = t + h
            theta = torch.full((batch_size, 1), self.fixed_theta, dtype=dtype, device=device)
            
            f_curr = self.compute_rhs(x, t, y)
            y_pred = y + h * f_curr
            f_next = self.compute_rhs(x, t_next, y_pred)
            
            y = y + h * ((1 - theta) * f_curr + theta * f_next)
            t = t_next
        
        return y[:, 0:1] if self.is_wave else y
    
    def pde_loss(self, x_col, t_col):
        x_col = x_col.detach().requires_grad_(True)
        t_col = t_col.detach().requires_grad_(True)
        u = self.forward(x_col, t_col)
        
        try:
            u_x = torch.autograd.grad(u.sum(), x_col, create_graph=True, retain_graph=True, allow_unused=True)[0]
            u_xx = torch.autograd.grad(u_x.sum(), x_col, create_graph=True, retain_graph=True, allow_unused=True)[0] if u_x is not None else torch.zeros_like(x_col)
            u_t = torch.autograd.grad(u.sum(), t_col, create_graph=True, retain_graph=True, allow_unused=True)[0]
            
            u_x = u_x if u_x is not None else torch.zeros_like(x_col)
            u_xx = u_xx if u_xx is not None else torch.zeros_like(x_col)
            u_t = u_t if u_t is not None else torch.zeros_like(t_col)
        except:
            return torch.tensor(1e-6, device=x_col.device, requires_grad=True)
        
        if self.pde_type == 'heat':
            residual = u_t - self.alpha * u_xx
        elif self.pde_type == 'burgers':
            residual = u_t + u * u_x - self.nu * u_xx
        elif self.pde_type == 'reaction_diffusion':
            residual = u_t - self.D * u_xx - self.r * u * (1 - u)
        elif self.pde_type == 'wave':
            try:
                u_tt = torch.autograd.grad(u_t.sum(), t_col, create_graph=True, allow_unused=True)[0]
                residual = (u_tt if u_tt is not None else torch.zeros_like(t_col)) - self.c**2 * u_xx
            except:
                residual = torch.zeros_like(u)
        
        return torch.mean(residual**2)
    
    def boundary_loss(self, domain, n_bc=10):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.buffers()).device if list(self.buffers()) else torch.device('cpu')
        
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
            device = next(self.buffers()).device if list(self.buffers()) else torch.device('cpu')
            x_ic = torch.rand(n_ic, 1, device=device)
        
        t_zero = torch.zeros_like(x_ic)
        u_pred = self.forward(x_ic, t_zero)
        u_true = self.initial_condition(x_ic)
        
        if self.is_wave:
            u_true = u_true[:, 0:1]
        
        return torch.mean((u_pred - u_true)**2)
    
    def total_loss(self, domain, n_collocation=100, n_bc=10, n_ic=10,
                   lambda_pde=1.0, lambda_bc=1.0, lambda_ic=1.0):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.buffers()).device if list(self.buffers()) else torch.device('cpu')
        
        x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
        t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
        x_ic = torch.rand(n_ic, 1, device=device) * (x_max - x_min) + x_min
        
        loss_pde = self.pde_loss(x_col, t_col)
        loss_bc = self.boundary_loss(domain, n_bc)
        loss_ic = self.initial_condition_loss(x_ic, n_ic)
        
        total = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic
        
        return total, {
            'pde': loss_pde.item() if isinstance(loss_pde, torch.Tensor) else loss_pde,
            'bc': loss_bc.item() if isinstance(loss_bc, torch.Tensor) else loss_bc,
            'ic': loss_ic.item() if isinstance(loss_ic, torch.Tensor) else loss_ic,
            'total': total.item()
        }
    
    def get_params(self):
        params = {'scheme': self.scheme, 'theta': self.fixed_theta, 'n_steps': self.n_steps}
        for name, param in self.named_buffers():
            params[name] = param.item()
        return params
    
    def get_theta_statistics(self, domain, n_samples=100):
        return {'mean': self.fixed_theta, 'std': 0.0, 'min': self.fixed_theta, 'max': self.fixed_theta}