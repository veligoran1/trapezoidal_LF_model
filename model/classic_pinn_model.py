import torch
import torch.nn as nn
import numpy as np


class ClassicalPINN(nn.Module):
    """Классическая PINN без численных схем"""
    
    def __init__(self, pde_type: str, input_dim: int = 2, hidden_dim: int = 6):
        super().__init__()
        self.pde_type = pde_type
        self.input_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_params()
    
    def _init_params(self):
        if self.pde_type == 'heat':
            self.register_buffer('alpha', torch.tensor(1.0))
        elif self.pde_type == 'wave':
            self.register_buffer('c', torch.tensor(1.0))
        elif self.pde_type == 'burgers':
            self.register_buffer('nu', torch.tensor(0.01/np.pi))
        elif self.pde_type == 'poisson':
            pass
        elif self.pde_type == 'reaction_diffusion':
            self.register_buffer('D', torch.tensor(0.01))
            self.register_buffer('r', torch.tensor(1.0))
    
    def forward(self, x):
        return self.net(x)
    
    def pde_loss(self, x):
        """Прямое вычисление остатка PDE без численных схем"""
        x = x.clone().requires_grad_(True)
        u = self.net(x)
        
        # Производные
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        
        if self.pde_type == 'poisson':
            u_x, u_y = grad[:, 0:1], grad[:, 1:2]
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
            u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1:2]
            
            x_coord, y_coord = x[:, 0:1], x[:, 1:2]
            rhs = -2 * np.pi**2 * torch.sin(np.pi * x_coord) * torch.sin(np.pi * y_coord)
            residual = u_xx + u_yy - rhs
            
        else:  # Эволюционные уравнения
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
        """Граничные и начальные условия (обновлено для включения IC)"""

        bc_loss = 0.0
        n_bc = 100
        
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
                    # Используем приближение, так как t_min мал, или точное
                    u_target = -torch.sin(np.pi * x_ic)  # Приближение для малого t_min
                    # Для точного: u_target = torch.from_numpy(compute_burgers_exact(points_ic.cpu(), self.nu.item())).to(points_ic.device)
                elif self.pde_type == 'reaction_diffusion':
                    u_target = 0.5 * (1 + torch.tanh(x_ic / np.sqrt(2 * self.D)))
                
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
        """Для совместимости"""
        params = {}
        for name, param in self.named_buffers():
            params[name] = param.item()
        return params