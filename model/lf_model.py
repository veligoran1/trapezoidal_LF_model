import torch
import torch.nn as nn
import numpy as np

class LearnableTrapezoidal(nn.Module):
    """Low-Fidelity PINN с обучаемым методом трапеций"""
    
    def __init__(self, pde_type: str, input_dim: int = 2, hidden_dim: int = 5):
        super().__init__()
        self.pde_type = pde_type
        self.input_dim = input_dim
        
        # Простая нейросеть
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Обучаемый параметр u ∈ [0,1] для метода трапеций
        self.u_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        # Шаг для численной схемы
        self.h = 0.01
        
        # Физические параметры
        self._init_params()
    
    def _init_params(self):
        """Задаем физические параметры для разных уравнений"""
        if self.pde_type == 'heat':
            self.register_buffer('alpha', torch.tensor(1))
        elif self.pde_type == 'wave':
            self.register_buffer('c', torch.tensor(1.0))
        elif self.pde_type == 'burgers':
            self.register_buffer('nu', torch.tensor(0.01/np.pi))
        elif self.pde_type == 'poisson':
            pass  # без параметров
        elif self.pde_type == 'reaction_diffusion':
            self.register_buffer('D', torch.tensor(0.01))
            self.register_buffer('r', torch.tensor(1.0))
    
    @property 
    def u(self):
        """Параметр трапеций u ∈ [0,1]"""
        return torch.sigmoid(self.u_raw)
    
    def forward(self, x):
        return self.net(x)
    
    def derivatives(self, x):
        """Вычисление производных"""
        x = x.clone().requires_grad_(True)
        u = self.net(x)
        
        # Первые производные
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d = {'u': u}
        
        if self.input_dim == 2:
            if self.pde_type == 'poisson':
                # Стационарное 2D: (x,y)
                d['u_x'], d['u_y'] = grad[:, 0:1], grad[:, 1:2]
                d['u_xx'] = torch.autograd.grad(d['u_x'].sum(), x, create_graph=True)[0][:, 0:1]
                d['u_yy'] = torch.autograd.grad(d['u_y'].sum(), x, create_graph=True)[0][:, 1:2]
                d['laplacian'] = d['u_xx'] + d['u_yy']
            else:
                # Эволюционные 1D + время: (x,t)
                d['u_x'], d['u_t'] = grad[:, 0:1], grad[:, 1:2]
                d['u_xx'] = torch.autograd.grad(d['u_x'].sum(), x, create_graph=True)[0][:, 0:1]
                
                # Для волнового уравнения нужна вторая производная по времени
                if self.pde_type == 'wave':
                    d['u_tt'] = torch.autograd.grad(d['u_t'].sum(), x, create_graph=True)[0][:, 1:2]
        
        return d
    
    def _get_rhs_function(self, d, x):
        """Правая часть PDE: f(x, u(x))"""
        if self.pde_type == 'heat':
            # ∂u/∂t = α∇²u
            return self.alpha * d['u_xx']
            
        elif self.pde_type == 'wave':
            # ∂²u/∂t² = c²∇²u
            return self.c**2 * d['u_xx']
            
        elif self.pde_type == 'burgers':
            # ∂u/∂t = -u∂u/∂x + ν∂²u/∂x²
            return -d['u'] * d['u_x'] + self.nu * d['u_xx']
            
        elif self.pde_type == 'poisson':
            # ∇²u = f(x,y)
            x_coord, y_coord = x[:, 0:1], x[:, 1:2]
            return -2 * np.pi**2 * torch.sin(np.pi * x_coord) * torch.sin(np.pi * y_coord)
            
        elif self.pde_type == 'reaction_diffusion':
            # ∂u/∂t = D∇²u + ru(1-u)
            return self.D * d['u_xx'] + self.r * d['u'] * (1 - d['u'])
        
        else:
            raise ValueError(f"Неизвестное PDE: {self.pde_type}")
    
    def pde_loss(self, x):
        """Обобщенный метод трапеций для PDE"""
        u_param = self.u
        h = self.h
        
        # Определяем по какой координате делать шаг
        if self.pde_type == 'poisson':
            coord_idx = 0  # x-координата для стационарного
        else:
            coord_idx = 1  # время для эволюционных
        
        # Создаем сдвинутые точки: x₀ + h
        x_shifted = x.clone()
        x_shifted[:, coord_idx:coord_idx+1] += h
        
        # Вычисляем производные в обеих точках
        d_current = self.derivatives(x)      # в точке x₀
        d_shifted = self.derivatives(x_shifted)  # в точке x₀ + h
        
        # Правые части в обеих точках
        f_current = self._get_rhs_function(d_current, x)      # f(x₀, u(x₀))
        f_shifted = self._get_rhs_function(d_shifted, x_shifted)  # f(x₀+h, u(x₀+h))
        
        # Обобщенная формула трапеций: (1-u)·f(x₀) + u·f(x₀+h)
        combined_rhs = (1 - u_param) * f_current + u_param * f_shifted
        
        # Левая часть уравнения (производная по соответствующей координате)
        if self.pde_type == 'heat':
            lhs = d_current['u_t']
        elif self.pde_type == 'wave':
            lhs = d_current['u_tt']
        elif self.pde_type == 'burgers':
            lhs = d_current['u_t']
        elif self.pde_type == 'poisson':
            lhs = d_current['laplacian']
        elif self.pde_type == 'reaction_diffusion':
            lhs = d_current['u_t']
        
        # Остаток уравнения: lhs = combined_rhs
        residual = lhs - combined_rhs
        return torch.mean(residual**2)
    
    def boundary_loss(self, domain):
        """Граничные условия для разных типов PDE"""
        bc_loss = 0.0
        n_bc = 100
        
        if self.input_dim == 2:
            if 't' in domain:  # 1D + время
                x_min, x_max = domain['x']
                t_min, t_max = domain['t']
                
                t_bc = torch.rand(n_bc, 1) * (t_max - t_min) + t_min
                x_left = torch.full_like(t_bc, x_min)
                x_right = torch.full_like(t_bc, x_max)
                points_left = torch.cat([x_left, t_bc], dim=1)
                points_right = torch.cat([x_right, t_bc], dim=1)
                
                if self.pde_type == 'burgers':
                    # Периодические граничные условия: u(x_min,t) = u(x_max,t)
                    bc_loss += torch.mean((self(points_left) - self(points_right))**2)
                    
                elif self.pde_type == 'reaction_diffusion':
                    # Бегущая волна: u=0 слева, u=1 справа
                    bc_loss += torch.mean(self(points_left)**2)
                    bc_loss += torch.mean((self(points_right) - 1.0)**2)
                    
                else:
                    # Dirichlet по умолчанию: u = 0 на границах
                    bc_loss += torch.mean(self(points_left)**2)
                    bc_loss += torch.mean(self(points_right)**2)
                    
            else:  # 2D стационарный (x,y)
                x_min, x_max = domain['x']
                y_min, y_max = domain['y']
                
                # Четыре стороны квадрата
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
        """Возвращает параметры модели"""
        params = {'u': self.u.item()}
        for name, param in self.named_buffers():
            params[name] = param.item()
        return params