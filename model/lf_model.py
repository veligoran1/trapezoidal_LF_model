import torch
import torch.nn as nn
import numpy as np

class LowFidelityPINN(nn.Module):
    
    def __init__(self, pde_type: str, n_steps: int = 10, theta_hidden_dim: int = 5, 
                 n_iterations: int = 2, lr: float = 0.001, max_h: float = 0.1):
        super().__init__()
        self.pde_type = pde_type
        self.n_steps = n_steps  # Максимальное количество шагов
        self.is_wave = (pde_type == 'wave')
        self.n_iterations = n_iterations  # Количество fixed-point итераций
        self.lr = lr  # Learning rate для оптимизатора
        self.max_h = max_h  # Максимальный размер одного временного шага
        
        self.theta_net = nn.Sequential(
            nn.Linear(4, theta_hidden_dim), nn.Tanh(),
            nn.Linear(theta_hidden_dim, 1), nn.Sigmoid()
        )
        
        self._init_params()
        
        with torch.no_grad():
            self.theta_net[-2].bias.fill_(0.0)

        self._print_info()
    
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
        print(f"\n{'='*60}")
        print(f"Low-Fidelity PINN (Fixed-Point + Adaptive Steps)")
        print(f"PDE: {self.pde_type} | Max Steps: {self.n_steps} | Params: {n_params}")
        print(f"Iterations: {self.n_iterations} | LR: {self.lr} | Max h: {self.max_h}")
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
    
    def forward(self, x, t_end):
        """
        Использует self.n_iterations для fixed-point итераций
        Адаптивно выбирает количество шагов на основе t_end
        """
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
        
        # ============================================================
        # АДАПТИВНЫЙ ВЫБОР n_steps
        # ============================================================
        # Вычисляем нужное количество шагов чтобы h <= self.max_h
        n_steps_adaptive = torch.ceil(t_end / self.max_h).max().int().item()
        n_steps_adaptive = max(1, min(n_steps_adaptive, self.n_steps))  # Ограничиваем self.n_steps
        
        h = t_end / n_steps_adaptive
        # ============================================================
        
        y = self.initial_condition(x)
        t = torch.zeros(batch_size, 1, dtype=dtype, device=device)
    
        for _ in range(n_steps_adaptive):
            t_next = t + h

            y_for_theta = y[:, 0:1] if self.is_wave else y

            # ДОБАВЛЕНО: Вычисляем grad_norm
            y_x = torch.autograd.grad(
                y_for_theta.sum(), x,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if y_x is None:
                y_x = torch.zeros_like(x)
            grad_norm = torch.abs(y_x)
            
            # Нормализуем grad_norm:
            grad_norm_norm = torch.tanh(grad_norm / 5.0)
            
            # ИЗМЕНЕНО: 5 входов с detach
            theta = self.theta_net(torch.cat([
                x.detach(),
                t.detach(),
                t_next.detach(),
                grad_norm_norm.detach()
            ], dim=1))

            t_next = t + h
            
            # Вычисляем f_curr один раз
            f_curr = self.compute_rhs(x, t, y)

            # Fixed-point итерации (используем self.n_iterations)
            y_new = y.clone()
            for iter in range(self.n_iterations):
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
        
        # Не нужны градиенты для forward в BC
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
        
        t_zero = torch.zeros_like(x_ic, device=x_ic.device)
        
        # Не нужны градиенты для forward в IC
        u_pred = self.forward(x_ic, t_zero)
        
        u_true = self.initial_condition(x_ic)
        
        if self.is_wave:
            u_true = u_true[:, 0:1]
        
        return torch.mean((u_pred - u_true)**2)
    
    def total_loss(self, domain, n_collocation=100,
                   lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        # Используем n_collocation для всего - проще и логичнее
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
        params = {'n_steps': self.n_steps, 'n_iterations': self.n_iterations, 'lr': self.lr}
        for name, param in self.named_buffers():
            params[name] = param.item()
        return params
    
    def get_optimizer(self):
        """Создает оптимизатор с learning rate из self.lr"""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_theta_statistics(self, domain, n_samples=100):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_sample = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
        t_sample = torch.rand(n_samples, 1, device=device) * (t_max - t_min) + t_min
        h = (t_max - t_min) / self.n_steps
        t_next = torch.clamp(t_sample + h, max=t_max)
        
        with torch.no_grad():
            # Используем IC как приближение y:
            y_approx = self.initial_condition(x_sample)
            if self.is_wave:
                y_approx = y_approx[:, 0:1]
            
            # Упрощение: grad_norm = 0 (для статистики OK)
            # ДОБАВЛЕНО: grad_norm (упрощённо = 0)
            grad_norm_approx = torch.zeros_like(x_sample)
            
            # ИЗМЕНЕНО: 5 входов
            theta_values = self.theta_net(torch.cat([
                x_sample,
                t_sample,
                t_next,
                grad_norm_approx
            ], dim=1))
        
        return {
            'mean': theta_values.mean().item(),
            'std': theta_values.std().item(),
            'min': theta_values.min().item(),
            'max': theta_values.max().item()
        }