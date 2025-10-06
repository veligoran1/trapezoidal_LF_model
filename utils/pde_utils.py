import torch
import numpy as np
from scipy.integrate import quad

# ============================================================
# ГЕНЕРАЦИЯ ТОЧЕК
# ============================================================

def generate_points(domain: dict, n_points: int):
    if 't' in domain:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        t = torch.rand(n_points, 1) * (domain['t'][1] - domain['t'][0]) + domain['t'][0]
        return torch.cat([x, t], dim=1)
    else:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        y = torch.rand(n_points, 1) * (domain['y'][1] - domain['y'][0]) + domain['y'][0]
        return torch.cat([x, y], dim=1)

def create_grid(domain: dict, n: int = 50):
    if 't' in domain:
        x = torch.linspace(domain['x'][0], domain['x'][1], n)
        t = torch.linspace(domain['t'][0], domain['t'][1], n)
        X, T = torch.meshgrid(x, t, indexing='ij')
        return torch.stack([X.flatten(), T.flatten()], dim=1), (n, n)
    else:
        x = torch.linspace(domain['x'][0], domain['x'][1], n)
        y = torch.linspace(domain['y'][0], domain['y'][1], n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([X.flatten(), Y.flatten()], dim=1), (n, n)

# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_universal(model, domain: dict, epochs: int = 2000, lr: float = 0.001, 
                   n_collocation: int = 100, n_bc: int = 50, n_ic: int = 50,
                   lambda_pde: float = 1.0, lambda_bc: float = 1.0, lambda_ic: float = 1.0):
    trainable_params = list(model.parameters())
    
    if not trainable_params:
        return {
            'losses': [0.0], 'pde_losses': [0.0], 'bc_losses': [0.0], 
            'ic_losses': [0.0], 'theta_statistics': [None],
            'params': [model.get_params() if hasattr(model, 'get_params') else {}]
        }
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        'losses': [], 'pde_losses': [], 'bc_losses': [], 
        'ic_losses': [], 'theta_statistics': [], 'params': []
    }
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if hasattr(model, 'total_loss'):
            total_loss, loss_dict = model.total_loss(
                domain, n_collocation, n_bc, n_ic, lambda_pde, lambda_bc, lambda_ic
            )
            pde_loss, bc_loss, ic_loss = loss_dict['pde'], loss_dict['bc'], loss_dict['ic']
        else:
            x_min, x_max = domain['x']
            t_min, t_max = domain['t']
            device = next(model.parameters()).device
            
            x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
            t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
            points_col = torch.cat([x_col, t_col], dim=1)
            pde_loss_val = model.pde_loss(points_col)
            
            bc_loss_val = model.boundary_loss(domain)
            
            x_ic = torch.rand(n_ic, 1, device=device) * (x_max - x_min) + x_min
            t_ic = torch.zeros_like(x_ic)
            points_ic = torch.cat([x_ic, t_ic], dim=1)
            u_pred_ic = model(points_ic)
            
            ic_conditions = {
                'heat': lambda x: torch.sin(np.pi * x),
                'wave': lambda x: torch.sin(np.pi * x),
                'burgers': lambda x: -torch.sin(np.pi * x),
                'reaction_diffusion': lambda x: 0.5 * (1 + torch.tanh(x / np.sqrt(2 * 0.01)))
            }
            u_true_ic = ic_conditions.get(model.pde_type, lambda x: torch.zeros_like(x))(x_ic)
            ic_loss_val = torch.mean((u_pred_ic - u_true_ic)**2)
            
            total_loss = lambda_pde * pde_loss_val + lambda_bc * bc_loss_val + lambda_ic * ic_loss_val
            pde_loss, bc_loss, ic_loss = pde_loss_val.item(), bc_loss_val.item(), ic_loss_val.item()
        
        if torch.isnan(total_loss):
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if hasattr(model, 'get_theta_statistics'):
            theta_stats = model.get_theta_statistics(domain)
        elif hasattr(model, 'fixed_theta'):
            theta_stats = {'mean': model.fixed_theta, 'std': 0.0, 'min': model.fixed_theta, 'max': model.fixed_theta}
        else:
            theta_stats = None
        
        params_dict = model.get_params() if hasattr(model, 'get_params') else {}
        
        history['losses'].append(total_loss.item())
        history['pde_losses'].append(pde_loss)
        history['bc_losses'].append(bc_loss)
        history['ic_losses'].append(ic_loss)
        history['theta_statistics'].append(theta_stats)
        history['params'].append(params_dict.copy() if params_dict else {})
        
        if epoch % max(1, epochs // 10) == 0:
            theta_display = f'θ={theta_stats["mean"]:.3f}±{theta_stats["std"]:.3f}' if theta_stats and theta_stats.get('std', 0) > 0 else f'θ={theta_stats["mean"]:.3f}' if theta_stats else 'Classical'
            print(f'   Epoch {epoch:4d}: Loss={total_loss.item():.2e}, PDE={pde_loss:.2e}, BC={bc_loss:.2e}, IC={ic_loss:.2e}, {theta_display}')
    
    return history

# ============================================================
# ОЦЕНКА
# ============================================================

def evaluate(model, domain: dict, exact_solution=None, n_test: int = 2500):
    model.eval()
    test_points, grid_shape = create_grid(domain, int(np.sqrt(n_test)))
    is_low_fidelity = hasattr(model, 'n_steps')
    
    if is_low_fidelity and 't' in domain:
        x_test, t_test = test_points[:, 0:1], test_points[:, 1:2]
        u_pred_list = []
        
        batch_size = 100
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:min(i+batch_size, len(x_test))].clone().requires_grad_(True)
            t_batch = t_test[i:min(i+batch_size, len(t_test))]
            with torch.set_grad_enabled(True):
                u_batch = model.forward(x_batch, t_batch)
            u_pred_list.append(u_batch.detach())
        
        u_pred = torch.cat(u_pred_list, dim=0)
    else:
        with torch.no_grad():
            u_pred = model(test_points)
    
    if hasattr(model, 'get_theta_statistics') and 't' in domain:
        theta_stats = model.get_theta_statistics(domain, n_samples=100)
    else:
        params = model.get_params() if hasattr(model, 'get_params') else {}
        theta_stats = {'mean': params.get('theta', 0), 'std': 0.0} if 'scheme' in params else None
    
    pde_residual = 0.0
    if hasattr(model, 'pde_loss'):
        try:
            with torch.enable_grad():
                # Используем случайную выборку вместо первых 20 точек для лучшей репрезентативности
                sample_idx = torch.randperm(len(test_points))[:50]
                if is_low_fidelity and 't' in domain:
                    pde_residual = model.pde_loss(
                        test_points[sample_idx, 0:1], 
                        test_points[sample_idx, 1:2]
                    ).item()
                else:
                    pde_residual = model.pde_loss(test_points[sample_idx]).item()
        except:
            pass
    
    results = {
        'points': test_points, 'u_pred': u_pred, 'grid_shape': grid_shape,
        'pde_type': model.pde_type, 'pde_residual': pde_residual,
        'theta_statistics': theta_stats, 'is_spatial_2d': 'y' in domain,
        'model_type': model.__class__.__name__
    }
    
    if exact_solution:
        with torch.no_grad():
            u_exact = exact_solution(test_points)
            if u_exact.dim() == 1:
                u_exact = u_exact.unsqueeze(1)
            error = torch.abs(u_pred - u_exact)
            results.update({
                'u_exact': u_exact, 'error': error,
                'max_error': torch.max(error).item(),
                'mean_error': torch.mean(error).item()
            })
    
    return results

# ============================================================
# ТОЧНЫЕ РЕШЕНИЯ
# ============================================================

def compute_burgers_exact(coords, nu):
    """Вычисление точного решения уравнения Бюргерса"""
    x, t = coords[:, 0:1].numpy(), coords[:, 1:2].numpy()
    f = lambda y: np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
    
    result = np.zeros_like(x)
    for i in range(len(x)):
        if t[i] < 1e-6:
            result[i] = -np.sin(np.pi * x[i])
        else:
            num_int = lambda eta: np.sin(np.pi * (x[i,0] - eta)) * f(x[i,0] - eta) * np.exp(-eta**2 / (4 * nu * t[i,0]))
            den_int = lambda eta: f(x[i,0] - eta) * np.exp(-eta**2 / (4 * nu * t[i,0]))
            num, _ = quad(num_int, -10, 10, limit=100)
            den, _ = quad(den_int, -10, 10, limit=100)
            result[i] = -num / den
    
    return torch.tensor(result, dtype=torch.float32)

def get_exact_solution(pde_type: str):
    """Получить точное решение для указанного типа PDE"""
    solutions = {
        'heat': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.exp(-np.pi**2 * c[:, 1:2]),
        'wave': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.cos(np.pi * c[:, 1:2]),
        'burgers': lambda c: compute_burgers_exact(c, nu=0.01/np.pi),
        'poisson': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.sin(np.pi * c[:, 1:2]),
        'reaction_diffusion': lambda c: 0.5 * (1 + torch.tanh((c[:, 0:1] - 0.5 * c[:, 1:2]) / np.sqrt(2 * 0.01)))
    }
    return solutions.get(pde_type, None)