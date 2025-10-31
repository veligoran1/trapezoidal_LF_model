import torch
import numpy as np
from scipy.integrate import quad
import time

# ============================================================
# ГЕНЕРАЦИЯ ТОЧЕК
# ============================================================

def generate_points(domain: dict, n_points: int):
    """Генерация случайных точек в области"""
    if 't' in domain:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        t = torch.rand(n_points, 1) * (domain['t'][1] - domain['t'][0]) + domain['t'][0]
        return torch.cat([x, t], dim=1)
    else:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        y = torch.rand(n_points, 1) * (domain['y'][1] - domain['y'][0]) + domain['y'][0]
        return torch.cat([x, y], dim=1)

def create_grid(domain: dict, n: int = 50):
    """Создание регулярной сетки для визуализации"""
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
                    n_collocation: int = 100, max_time: float = None, 
                    target_metric: str = 'l2re', target_value: float = None, 
                    eval_interval: int = 5):
    """
    Универсальная функция обучения для LF-PINN и Classical PINN.
    Добавлены: max_time (секунды) для варианта 1.
    Для варианта 2: target_metric может быть 'loss', 'l2re' или 'rmse', с target_value.
    Если metric 'l2re' или 'rmse', вызываем evaluate каждые eval_interval эпох (overhead!).
    """
    trainable_params = list(model.parameters())
    
    if not trainable_params:
        return {
            'losses': [0.0], 'pde_losses': [0.0], 
            'theta_statistics': [None],
            'params': [model.get_params() if hasattr(model, 'get_params') else {}],
            'epochs_completed': 0,
            'converged': False
        }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    history = {
        'losses': [], 'pde_losses': [], 
        'theta_statistics': [], 'params': [],
        'epochs_completed': 0,
        'converged': False
    }
    
    is_low_fidelity = hasattr(model, 'n_steps')
    exact_sol = get_exact_solution(model.pde_type)  # Для evaluate, если нужно
    
    start_train_time = time.time()
    converged = False
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if hasattr(model, 'total_loss'):
            total_loss, loss_dict = model.total_loss(domain, n_collocation)
            pde_loss = loss_dict['pde']
        else:
            x_min, x_max = domain['x']
            t_min, t_max = domain['t']
            device = next(model.parameters()).device
            
            x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
            t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
            points_col = torch.cat([x_col, t_col], dim=1)
            
            pde_loss_val = model.pde_loss(points_col)
            bc_ic_loss = model.boundary_loss(domain)
            total_loss = pde_loss_val + bc_ic_loss
            pde_loss = pde_loss_val.item()
        
        if torch.isnan(total_loss):
            print(f"   WARNING: NaN loss at epoch {epoch}, skipping...")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        theta_stats = model.get_theta_statistics(domain) if hasattr(model, 'get_theta_statistics') else None
        params_dict = model.get_params() if hasattr(model, 'get_params') else {}
        
        history['losses'].append(total_loss.item())
        history['pde_losses'].append(pde_loss)
        history['theta_statistics'].append(theta_stats)
        history['params'].append(params_dict.copy() if params_dict else {})
        
        if epoch % max(1, epochs // 10) == 0:
            theta_display = f'θ={theta_stats["mean"]:.3f}±{theta_stats["std"]:.3f}' if theta_stats and theta_stats.get('std', 0) > 1e-6 else f'θ={theta_stats["mean"]:.3f}' if theta_stats else 'Classical PINN'
            print(f'   Epoch {epoch:4d}: Loss={total_loss.item():.2e}, PDE={pde_loss:.2e}, {theta_display}')
        
        # Вариант 1: Остановка по времени
        elapsed_time = time.time() - start_train_time
        if max_time is not None and elapsed_time > max_time:
            print(f"   Training stopped after {epoch+1} epochs due to time limit ({max_time}s). Elapsed: {elapsed_time:.2f}s")
            break
        
        # Вариант 2: Проверка сходимости по выбранной метрике
        if target_value is not None:
            if target_metric == 'loss':
                current_val = total_loss.item()
                if current_val < target_value:
                    print(f"   Converged by {target_metric} < {target_value} at epoch {epoch+1} with value {current_val:.2e}. Time: {elapsed_time:.2f}s")
                    converged = True
                    break
            elif target_metric in ['l2re', 'rmse']:
                if (epoch + 1) % eval_interval == 0:
                    res = evaluate(model, domain, exact_solution=exact_sol)
                    current_val = res['l2re'] if target_metric == 'l2re' else res['rmse']
                    print(f"   Eval at epoch {epoch+1}: {target_metric.upper()}={current_val:.2e}")
                    if current_val < target_value:
                        print(f"   Converged by {target_metric} < {target_value} at epoch {epoch+1} with value {current_val:.2e}. Time: {elapsed_time:.2f}s")
                        converged = True
                        break
    
    history['epochs_completed'] = epoch + 1
    history['converged'] = converged
    history['training_time'] = time.time() - start_train_time  # Всегда сохраняем реальное время
    
    return history

# ============================================================
# ОЦЕНКА
# ============================================================

def evaluate(model, domain: dict, exact_solution=None, n_test: int = 2500):
    """
    Оценка модели на тестовой сетке.
    Совместима с LF-PINN и Classical PINN.
    """
    model.eval()
    test_points, grid_shape = create_grid(domain, int(np.sqrt(n_test)))
    is_low_fidelity = hasattr(model, 'n_steps')
    
    # Предсказание в зависимости от типа модели
    if is_low_fidelity and 't' in domain:
        # LF-PINN требует отдельные x и t
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
        # Classical PINN просто передаем точки
        with torch.no_grad():
            u_pred = model(test_points)
    
    # Сбор статистики θ
    if hasattr(model, 'get_theta_statistics') and 't' in domain:
        theta_stats = model.get_theta_statistics(domain, n_samples=100)
    else:
        theta_stats = None
    
    # Вычисление PDE residual на подвыборке точек
    pde_residual = 0.0
    if hasattr(model, 'pde_loss'):
        try:
            with torch.enable_grad():
                sample_idx = torch.randperm(len(test_points))[:50]
                if is_low_fidelity and 't' in domain:
                    pde_residual = model.pde_loss(
                        test_points[sample_idx, 0:1], 
                        test_points[sample_idx, 1:2]
                    ).item()
                else:
                    pde_residual = model.pde_loss(test_points[sample_idx]).item()
        except Exception as e:
            print(f"   WARNING: Could not compute PDE residual: {e}")
            pde_residual = 0.0
    
    # Базовые результаты
    results = {
        'points': test_points, 
        'u_pred': u_pred, 
        'grid_shape': grid_shape,
        'pde_type': model.pde_type, 
        'pde_residual': pde_residual,
        'theta_statistics': theta_stats, 
        'is_spatial_2d': 'y' in domain,
        'model_type': model.__class__.__name__
    }
    
    # Если есть точное решение, вычисляем ошибки
    if exact_solution:
        with torch.no_grad():
            u_exact = exact_solution(test_points)
            if u_exact.dim() == 1:
                u_exact = u_exact.unsqueeze(1)
            error = torch.abs(u_pred - u_exact)
            results.update({
                'u_exact': u_exact, 
                'error': error,
                'mean_error': torch.mean(error).item(),
                'max_err': torch.max(error).item(),
                'rmse': torch.sqrt(torch.mean(error**2)).item(),
                'l2re': (torch.sqrt(torch.mean(error**2)) / torch.sqrt(torch.mean(u_exact**2) + 1e-8)).item()
            })
    
    return results

# ============================================================
# ТОЧНЫЕ РЕШЕНИЯ
# ============================================================

def compute_burgers_exact(coords, nu):
    """
    Точное решение для Burgers equation через Cole-Hopf преобразование.
    
    PDE: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    IC:  u(x,0) = -sin(π(x + 0.5)) на x ∈ [-0.5, 1.0]
    
    Решение:
        u(x,t) = -2ν·∂φ/∂x / φ
    где
        φ(x,t) = ∫_{-∞}^{∞} exp(-(x-ξ)²/(4νt) - F(ξ)/(2ν)) dξ
        F(ξ) = ∫₀^ξ u₀(s) ds = ∫₀^ξ -sin(π(s+0.5)) ds = [cos(π(s+0.5))/π]₀^ξ
             = (cos(π(ξ+0.5)) - cos(π·0.5)) / π
             = (cos(π(ξ+0.5)) - 0) / π  (если 0.5 сдвиг)
    
    НО! Нужно учесть что F(ξ) — первообразная от u₀:
        F'(ξ) = u₀(ξ) = -sin(π(ξ+0.5))
        F(ξ) = cos(π(ξ+0.5))/π + C
    """
    from scipy.integrate import quad
    import numpy as np
    import torch
    
    x, t = coords[:, 0:1].numpy(), coords[:, 1:2].numpy()
    
    # Первообразная от IC:
    def F_primitive(xi):
        """∫ u₀(s) ds = ∫ -sin(π(s+0.5)) ds"""
        return np.cos(np.pi * (xi)) / np.pi
    
    result = np.zeros_like(x)
    
    for i in range(len(x)):
        xi = x[i, 0]
        ti = t[i, 0]
        
        # При t≈0 возвращаем IC:
        if ti < 1e-8:
            result[i] = -np.sin(np.pi * xi)
            continue
        
        # Адаптивные пределы интегрирования:
        sigma = np.sqrt(4 * nu * ti)  # Стандартное отклонение гауссиана
        limit_low = xi - 6 * sigma    # -6σ
        limit_high = xi + 6 * sigma   # +6σ
        
        # Ограничим пределами домена (с запасом):
        limit_low = max(limit_low, -5.0)
        limit_high = min(limit_high, 5.0)
        
        # Интегралы для числителя и знаменателя:
        def integrand_phi(eta):
            """Подынтегральное выражение для φ"""
            gauss = np.exp(-(xi - eta)**2 / (4 * nu * ti))
            hopf = np.exp(-F_primitive(eta) / (2 * nu))
            return gauss * hopf
        
        def integrand_phi_x(eta):
            """Подынтегральное выражение для ∂φ/∂x"""
            gauss_deriv = -(xi - eta) / (2 * nu * ti) * np.exp(-(xi - eta)**2 / (4 * nu * ti))
            hopf = np.exp(-F_primitive(eta) / (2 * nu))
            return gauss_deriv * hopf
        
        try:
            # Вычисляем φ и ∂φ/∂x:
            phi, _ = quad(integrand_phi, limit_low, limit_high, 
                         limit=100, epsabs=1e-10, epsrel=1e-8)
            phi_x, _ = quad(integrand_phi_x, limit_low, limit_high, 
                           limit=100, epsabs=1e-10, epsrel=1e-8)
            
            # Cole-Hopf формула:
            if abs(phi) > 1e-12:
                result[i] = -2 * nu * phi_x / phi
            else:
                # Fallback при малых phi:
                result[i] = -np.sin(np.pi * (xi))
        
        except Exception as e:
            print(f"Warning: Integration failed at x={xi:.3f}, t={ti:.3f}: {e}")
            result[i] = -np.sin(np.pi * (xi))
    
    return torch.tensor(result, dtype=coords.dtype, device=coords.device)


def get_exact_solution_burgers(nu=0.01):
    """Wrapper для точного решения Burgers"""
    def solution(coords):
        return compute_burgers_exact(coords, nu)
    return solution

def get_exact_solution(pde_type: str):
    """Получить точное решение для указанного типа PDE"""
    solutions = {
        'heat': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.exp(-np.pi**2 * c[:, 1:2]),
        'wave': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.cos(np.pi * c[:, 1:2]),
        'burgers': lambda c: compute_burgers_exact(c, nu=0.01),
        'poisson': lambda c: torch.sin(np.pi * c[:, 0:1]) * torch.sin(np.pi * c[:, 1:2]),
        'reaction_diffusion': lambda c: 0.5 * (1 + torch.tanh((c[:, 0:1] - 0.2 * c[:, 1:2]) / np.sqrt(2 * 0.01)))
    }
    return solutions.get(pde_type, None)