import torch
import torch.nn as nn
import numpy as np


class ClassicalPINN(nn.Module):
    """Классическая PINN без численных схем"""
    
    def __init__(self, pde_type: str, input_dim: int = 2, hidden_dim: int = 6,
                learnable_params: bool = False, param_init: float = 0.5):
        super().__init__()
        self.pde_type = pde_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # ДОБАВЛЕНО: сохраняем hidden_dim
        
        self.learnable_params = learnable_params
        self.param_init = param_init

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_params()
    
    def _init_params(self):
        """Инициализация параметров PDE (buffer или Parameter)"""
        init_val = self.param_init if self.learnable_params else None
        
        if self.pde_type == 'heat':
            val = init_val if self.learnable_params else 1.0
            if self.learnable_params:
                self.alpha = nn.Parameter(torch.tensor(val))
            else:
                self.register_buffer('alpha', torch.tensor(val))
                
        elif self.pde_type == 'wave':
            val = init_val if self.learnable_params else 1.0
            if self.learnable_params:
                self.c = nn.Parameter(torch.tensor(val))
            else:
                self.register_buffer('c', torch.tensor(val))
                
        elif self.pde_type == 'burgers':
            val = init_val if self.learnable_params else 0.01
            if self.learnable_params:
                self.nu = nn.Parameter(torch.tensor(val))
            else:
                self.register_buffer('nu', torch.tensor(val))
            
        elif self.pde_type == 'reaction_diffusion':
            val_D = init_val if self.learnable_params else 0.01
            val_r = init_val if self.learnable_params else 1.0
            if self.learnable_params:
                self.D = nn.Parameter(torch.tensor(val_D))
                self.r = nn.Parameter(torch.tensor(val_r))
            else:
                self.register_buffer('D', torch.tensor(val_D))
                self.register_buffer('r', torch.tensor(val_r))
    
    def forward(self, x):
        return self.net(x)
    
    def pde_loss(self, x):
        """Прямое вычисление остатка PDE без численных схем"""
        x = x.clone().requires_grad_(True)
        u = self.net(x)
        
        # Производные
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            
        u_x, u_t = grad[:, 0:1], grad[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
            
        if self.pde_type == 'heat':
            residual = u_t - self.alpha * u_xx
        elif self.pde_type == 'wave':
            u_tt = torch.autograd.grad(u_t.sum(), x, create_graph=True)[0][:, 1:2]
            residual = u_tt - self.c**2 * u_xx
        elif self.pde_type == 'burgers':
            residual = u_t + u * u_x - self.nu * u_xx
        elif self.pde_type == 'reaction_diffusion':
            residual = u_t - self.D * u_xx - self.r * u * (1 - u)
        
        return torch.mean(residual**2)
    
    def boundary_loss(self, domain):
        """Граничные и начальные условия"""
        bc_loss = 0.0
        n_bc = 10
        
        if self.input_dim == 2:
            if 't' in domain:  # 1D + время
                x_min, x_max = domain['x']
                t_min, t_max = domain['t']
                
                # Пространственные границы
                t_bc = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
                x_left = torch.full_like(t_bc, x_min)
                x_right = torch.full_like(t_bc, x_max)
                points_left = torch.cat([x_left, t_bc], dim=1)
                points_right = torch.cat([x_right, t_bc], dim=1)
                
                if self.pde_type == 'burgers':
                    bc_loss += torch.mean((self(points_left) - self(points_right))**2)
                elif self.pde_type == 'reaction_diffusion':
                    bc_loss += torch.mean(self(points_left)**2)
                    bc_loss += torch.mean((self(points_right) - 1.0)**2)
                else:
                    bc_loss += torch.mean(self(points_left)**2)
                    bc_loss += torch.mean(self(points_right)**2)
                
                # Начальные условия на t = t_min
                x_ic = torch.rand(n_bc, 1) * (x_max - x_min) + x_min
                t_ic = torch.full_like(x_ic, t_min)
                points_ic = torch.cat([x_ic, t_ic], dim=1)
                
                if self.pde_type == 'heat':
                    u_target = torch.sin(np.pi * x_ic)
                elif self.pde_type == 'wave':
                    u_target = torch.sin(np.pi * x_ic)
                elif self.pde_type == 'burgers':
                    u_target = -torch.sin(np.pi * x_ic)
                elif self.pde_type == 'reaction_diffusion':
                    u_target = 0.5 * (1 + torch.tanh(x_ic / torch.sqrt(2 * self.D)))
                
                points_ic = points_ic.clone().requires_grad_(True)
                u_pred = self(points_ic)
                
                bc_loss += torch.mean((u_pred - u_target)**2)
                
                if self.pde_type == 'wave':
                    grad = torch.autograd.grad(u_pred.sum(), points_ic, create_graph=True)[0]
                    u_t = grad[:, 1:2]
                    bc_loss += torch.mean(u_t**2)
                    
            else:  # 2D стационарный
                x_min, x_max = domain['x']
                y_min, y_max = domain['y']
                
                sides = [
                    (torch.rand(n_bc, 1) * (x_max - x_min) + x_min, torch.full((n_bc, 1), y_min)),
                    (torch.rand(n_bc, 1) * (x_max - x_min) + x_min, torch.full((n_bc, 1), y_max)),
                    (torch.full((n_bc, 1), x_min), torch.rand(n_bc, 1) * (y_max - y_min) + y_min),
                    (torch.full((n_bc, 1), x_max), torch.rand(n_bc, 1) * (y_max - y_min) + y_min),
                ]
                
                for x_coords, y_coords in sides:
                    points = torch.cat([x_coords, y_coords], dim=1)
                    bc_loss += torch.mean(self(points)**2)
        
        return bc_loss
    
    def get_params(self):
        """Для совместимости и мониторинга"""
        params = {
            'hidden_dim': self.hidden_dim,
            'learnable_params': self.learnable_params
        }
        # Buffers
        for name, buf in self.named_buffers():
            if name in ['alpha', 'c', 'nu', 'D', 'r']:
                params[name] = buf.item()
        # Parameters (обучаемые)
        for name, p in self.named_parameters():
            if name in ['alpha', 'c', 'nu', 'D', 'r']:
                params[name] = p.item()
        return params
    
    def count_parameters(self):
        """Подсчет количества параметров"""
        return sum(p.numel() for p in self.parameters())