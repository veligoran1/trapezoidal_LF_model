import torch
import numpy as np
from scipy.integrate import quad

def generate_points(domain: dict, n_points: int):
    """Генерация случайных точек в домене"""
    if len(domain) == 2:
        if 't' in domain:  # 1D + время
            x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
            t = torch.rand(n_points, 1) * (domain['t'][1] - domain['t'][0]) + domain['t'][0]
            return torch.cat([x, t], dim=1)
        else:  # 2D стационарный
            x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
            y = torch.rand(n_points, 1) * (domain['y'][1] - domain['y'][0]) + domain['y'][0]
            return torch.cat([x, y], dim=1)

def generate_points_with_step(domain: dict, n_points: int, h: float):
    """
    Генерация точек с учетом шага h для метода трапеций
    Убираем точки слишком близко к границе времени/пространства
    """
    if len(domain) == 2:
        if 't' in domain:  # 1D + время
            x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
            # Время ограничиваем: [t_min, t_max - h]
            t_safe = max(domain['t'][1] - h, domain['t'][0] + 0.1)
            t = torch.rand(n_points, 1) * (t_safe - domain['t'][0]) + domain['t'][0]
            return torch.cat([x, t], dim=1)
        else:  # 2D стационарный
            # Пространство ограничиваем: [x_min, x_max - h]
            x_safe = max(domain['x'][1] - h, domain['x'][0] + 0.1)
            x = torch.rand(n_points, 1) * (x_safe - domain['x'][0]) + domain['x'][0]
            y = torch.rand(n_points, 1) * (domain['y'][1] - domain['y'][0]) + domain['y'][0]
            return torch.cat([x, y], dim=1)

def create_grid(domain: dict, n: int = 50):
    """Создание регулярной сетки"""
    if len(domain) == 2:
        if 't' in domain:  # 1D + время
            x = torch.linspace(domain['x'][0], domain['x'][1], n)
            t = torch.linspace(domain['t'][0], domain['t'][1], n)
            X, T = torch.meshgrid(x, t, indexing='ij')
            coords = torch.stack([X.flatten(), T.flatten()], dim=1)
            return coords, (n, n)
        else:  # 2D стационарный
            x = torch.linspace(domain['x'][0], domain['x'][1], n)
            y = torch.linspace(domain['y'][0], domain['y'][1], n)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
            return coords, (n, n)

def train(model, domain: dict, epochs: int = 2000, lr: float = 0.001, 
          n_points: int = 1000, exact_solution=None, 
          lambda_pde: float = 1.0, lambda_bc: float = 10.0, lambda_ic: float = 10.0):
    """
    Функция обучения для Low-Fidelity PINN с правильными весами
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'losses': [],
        'pde_losses': [],
        'bc_losses': [],
        'ic_losses': [], 
        'u_values': [],
        'params': []
    }
    
    # Определяем начальное условие (если эволюционное уравнение)
    has_time = 't' in domain
    if has_time:
        # Начальные условия для разных PDE
        initial_conditions = {
            'heat': lambda x: torch.sin(np.pi * x),
            'wave': lambda x: torch.sin(np.pi * x),
            'burgers': lambda x: -torch.sin(np.pi * x),
            'reaction_diffusion': lambda x: 0.5 * (1 + torch.tanh(x / np.sqrt(2 * model.D.item() / model.r.item())))
        }
        ic_func = initial_conditions.get(model.pde_type, None)
    else:
        ic_func = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Генерируем точки с учетом шага h
        points = generate_points_with_step(domain, n_points, model.h)
        
        # 1. PDE Loss
        pde_loss = model.pde_loss(points)
        
        # 2. Boundary Loss
        bc_loss = model.boundary_loss(domain)
        
        # 3. Initial Condition Loss (если есть время)
        ic_loss = 0.0
        if ic_func is not None and has_time:
            n_ic = n_points // 4
            x_ic = torch.rand(n_ic, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
            t_ic = torch.full_like(x_ic, domain['t'][0])  # t = t_min
            points_ic = torch.cat([x_ic, t_ic], dim=1)
            
            u_pred_ic = model(points_ic)
            u_true_ic = ic_func(x_ic)
            ic_loss = torch.mean((u_pred_ic - u_true_ic)**2)
        
        # Общая потеря с весами
        total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss + lambda_ic * ic_loss
        
        # Проверка на NaN
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss at epoch {epoch}")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Сохраняем историю
        u_val = model.u.item()
        history['losses'].append(total_loss.item())
        history['pde_losses'].append(pde_loss.item())
        history['bc_losses'].append(bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss)
        history['ic_losses'].append(ic_loss.item() if isinstance(ic_loss, torch.Tensor) else ic_loss)
        history['u_values'].append(u_val)
        history['params'].append(model.get_params().copy())
        
        # Вывод прогресса
        if epoch % (epochs//10) == 0:
            ic_str = f'IC={ic_loss:.2e}, ' if isinstance(ic_loss, torch.Tensor) else ''
            print(f'Epoch {epoch:4d}: '
                  f'Loss={total_loss:.2e}, '
                  f'PDE={pde_loss:.2e}, '
                  f'BC={bc_loss:.2e}, '
                  f'{ic_str}')
    
    return history

def evaluate(model, domain: dict, exact_solution=None, n_test: int = 2500):
    """Оценка модели"""
    model.eval()
    
    # Создаем регулярную сетку
    test_points, grid_shape = create_grid(domain, int(np.sqrt(n_test)))
    
    # Предсказание без градиентов
    with torch.no_grad():
        u_pred = model(test_points)
    
    # PDE residual нужен с градиентами
    pde_residual = model.pde_loss(test_points).item()
    
    results = {
        'points': test_points,
        'u_pred': u_pred,
        'grid_shape': grid_shape,
        'pde_residual': pde_residual,
        'u_param': model.u.item(),
        'is_spatial_2d': 'y' in domain
        }
    
    # Если есть точное решение
    if exact_solution:
        with torch.no_grad():
            u_exact = exact_solution(test_points)
            if u_exact.dim() == 1:
                u_exact = u_exact.unsqueeze(1)
            
            error = torch.abs(u_pred - u_exact)
            results.update({
                'u_exact': u_exact,
                'error': error,
                'max_error': torch.max(error).item(),
                'mean_error': torch.mean(error).item()
            })
    
    return results

def compute_burgers_exact(coords, nu):
    """
    Точное решение Бюргерса методом Cole-Hopf через интегралы
    """
    x = coords[:, 0:1].numpy()
    t = coords[:, 1:2].numpy()
    
    def f(y):
        return np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    
    def numerator_integrand(eta, x_val, t_val):
        return np.sin(np.pi * (x_val - eta)) * f(x_val - eta) * \
               np.exp(-eta**2 / (4 * nu * t_val))
    
    def denominator_integrand(eta, x_val, t_val):
        return f(x_val - eta) * np.exp(-eta**2 / (4 * nu * t_val))
    
    result = np.zeros_like(x)
    for i in range(len(x)):
        if t[i] < 1e-6:  # При t≈0 используем IC
            result[i] = -np.sin(np.pi * x[i])
        else:
            num, _ = quad(lambda eta: numerator_integrand(eta, x[i,0], t[i,0]), 
                         -10, 10, limit=100)
            den, _ = quad(lambda eta: denominator_integrand(eta, x[i,0], t[i,0]), 
                         -10, 10, limit=100)
            result[i] = -num / den
    
    return torch.tensor(result, dtype=torch.float32)

# Точные аналитические решения для тестирования
def get_exact_solution(pde_type: str):
    """Простые точные решения для основных PDE"""
    solutions = {
        'heat': lambda coords: torch.sin(np.pi * coords[:, 0:1]) * 
                              torch.exp(-np.pi**2 * 1 * coords[:, 1:2]),
    
        'wave': lambda coords: torch.sin(np.pi * coords[:, 0:1]) * 
                              torch.cos(np.pi * coords[:, 1:2]),
                              
        'burgers': lambda coords: compute_burgers_exact(coords, nu=0.01/np.pi),

        'poisson': lambda coords: torch.sin(np.pi * coords[:, 0:1]) * 
                                 torch.sin(np.pi * coords[:, 1:2]),
        
       'reaction_diffusion': lambda coords: 0.5 * (1 + torch.tanh(
                            (coords[:, 0:1] - (np.sqrt(0.01) * 1.0 / 2) * coords[:, 1:2]) / np.sqrt(2 * 0.01)))
    }
    
    return solutions.get(pde_type, None)