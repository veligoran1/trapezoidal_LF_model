import torch
import numpy as np
from scipy.integrate import quad
import time


# ============================================================
# POINT GENERATION
# ============================================================

def generate_points(domain: dict, n_points: int):
    """Generate random points in the domain"""
    if 't' in domain:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        t = torch.rand(n_points, 1) * (domain['t'][1] - domain['t'][0]) + domain['t'][0]
        return torch.cat([x, t], dim=1)
    else:
        x = torch.rand(n_points, 1) * (domain['x'][1] - domain['x'][0]) + domain['x'][0]
        y = torch.rand(n_points, 1) * (domain['y'][1] - domain['y'][0]) + domain['y'][0]
        return torch.cat([x, y], dim=1)

def create_grid(domain: dict, n: int = 50):
    """Create regular grid for visualization"""
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
# TRAINING
# ============================================================

def train_universal(model, domain: dict, epochs: int = 2000, lr: float = 0.001, 
                    n_collocation: int = 30, max_time: float = None, 
                    target_metric: str = 'l2re', target_value: float = None, 
                    eval_interval: int = 5):
    """
    Universal training function for LF-PINN and Classical PINN.
    Added: max_time (seconds) for option 1.
    For option 2: target_metric can be 'loss', 'l2re' or 'rmse', with target_value.
    If metric is 'l2re' or 'rmse', we call evaluate every eval_interval epochs (overhead!).
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        'losses': [], 'pde_losses': [], 
        'theta_statistics': [], 'params': [],
        'epochs_completed': 0,
        'converged': False
    }
    
    is_low_fidelity = hasattr(model, 'n_steps')
    exact_sol = get_exact_solution_parametric(model.pde_type)  # For evaluate, if needed
    
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
        
        # Option 1: Stop by time
        elapsed_time = time.time() - start_train_time
        if max_time is not None and elapsed_time > max_time:
            print(f"   Training stopped after {epoch+1} epochs due to time limit ({max_time}s). Elapsed: {elapsed_time:.2f}s")
            break
        
        # Option 2: Check convergence by selected metric
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
    history['training_time'] = time.time() - start_train_time  # Always save real time
    
    return history

def train_with_data(model, domain, data_points, data_values, 
                    epochs=2000, lr=0.001, n_collocation=30,
                    lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0, lambda_data=10.0,
                    max_time=None):
    """
    Train model with observation data
    
    Args:
        model: LowFidelityPINN or ClassicalPINN
        domain: dict with boundaries
        data_points: torch.Tensor [n_data, 2] - observation coordinates (x, t)
        data_values: torch.Tensor [n_data, 1] - measured values u
        epochs: number of epochs
        lr: learning rate
        n_collocation: number of collocation points
        lambda_pde, lambda_bc, lambda_ic, lambda_data: loss component weights
        max_time: maximum training time (seconds)
    """
    # Move data to model device
    device = next(model.parameters()).device
    data_points = data_points.to(device)
    data_values = data_values.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'losses': [], 'pde_losses': [], 'bc_losses': [], 
        'ic_losses': [], 'data_losses': [],
        'theta_statistics': [], 'params': [],
        'epochs_completed': 0,
        'training_time': 0
    }
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training with data:")
    print(f"  Number of observations: {len(data_points)}")
    print(f"  Lambda data: {lambda_data}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss (use model method if available, otherwise manually)
        if hasattr(model, 'total_loss_with_data'):
            total_loss, loss_dict = model.total_loss_with_data(
                domain, data_points, data_values, n_collocation,
                lambda_pde, lambda_bc, lambda_ic, lambda_data
            )
        else:
            # For Classical PINN - manually add data loss
            x_min, x_max = domain['x']
            t_min, t_max = domain['t']
            
            x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
            t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
            points_col = torch.cat([x_col, t_col], dim=1)
            
            pde_loss = model.pde_loss(points_col)
            bc_loss = model.boundary_loss(domain)
            
            # Data loss for Classical PINN
            u_pred = model(data_points)
            data_loss = torch.mean((u_pred - data_values)**2)
            
            total_loss = (lambda_pde * pde_loss + 
                         lambda_bc * bc_loss + 
                         lambda_data * data_loss)
            
            loss_dict = {
                'pde': pde_loss.item(),
                'bc': bc_loss.item(),
                'ic': 0.0,
                'data': data_loss.item(),
                'total': total_loss.item()
            }
        
        if torch.isnan(total_loss):
            print(f"   WARNING: NaN loss at epoch {epoch}, stopping...")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Save history
        history['losses'].append(loss_dict['total'])
        history['pde_losses'].append(loss_dict['pde'])
        history['bc_losses'].append(loss_dict['bc'])
        history['ic_losses'].append(loss_dict['ic'])
        history['data_losses'].append(loss_dict['data'])
        
        if hasattr(model, 'get_theta_statistics'):
            theta_stats = model.get_theta_statistics(domain)
            history['theta_statistics'].append(theta_stats)
        else:
            history['theta_statistics'].append(None)
        
        if hasattr(model, 'get_params'):
            history['params'].append(model.get_params())
        else:
            history['params'].append({})
        
        # Progress output
        if epoch % max(1, epochs // 10) == 0:
            print(f"   Epoch {epoch:4d}: Total={loss_dict['total']:.2e}, "
                  f"PDE={loss_dict['pde']:.2e}, Data={loss_dict['data']:.2e}")
        
        # Time check
        if max_time is not None:
            elapsed = time.time() - start_time
            if elapsed > max_time:
                print(f"   Training stopped due to time limit ({max_time}s)")
                break
    
    history['epochs_completed'] = epoch + 1
    history['training_time'] = time.time() - start_time
    
    return history

# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, domain: dict, exact_solution=None, n_test: int = 2500):
    """
    Evaluate model on test grid.
    Compatible with LF-PINN and Classical PINN.
    """
    model.eval()
    test_points, grid_shape = create_grid(domain, int(np.sqrt(n_test)))
    is_low_fidelity = hasattr(model, 'n_steps')
    
    # Prediction depending on model type
    if is_low_fidelity and 't' in domain:
        # LF-PINN requires separate x and t
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
        # Classical PINN just pass points
        with torch.no_grad():
            u_pred = model(test_points)
    
    # Collect θ statistics
    if hasattr(model, 'get_theta_statistics') and 't' in domain:
        theta_stats = model.get_theta_statistics(domain, n_samples=100)
    else:
        theta_stats = None
    
    # Compute PDE residual on subset of points
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
    
    # Base results
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
    
    # If exact solution available, compute errors
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

def evaluate_and_compare(models_dict, domain, pde_type, true_params=None):
    """
    Evaluate multiple models and compare.
    
    models_dict: {'name': model, ...}
    """    
    if true_params:
        exact_sol = get_exact_solution_parametric(pde_type, **true_params)
    else:
        exact_sol = get_exact_solution_parametric(pde_type)
    
    results = {}
    for name, model in models_dict.items():
        model.eval()
        res = evaluate(model, domain, exact_solution=exact_sol)
        results[name] = {
            'l2re': res.get('l2re', None),
            'rmse': res.get('rmse', None),
            'max_error': res.get('max_err', None)
        }
        print(f"{name:30s}: L2RE={res.get('l2re', 0):.4e}, RMSE={res.get('rmse', 0):.4e}")
    
    return results


# ============================================================
# EXACT SOLUTIONS
# ============================================================

def compute_burgers_exact(coords, nu):
    """
    Exact solution for Burgers equation via Cole-Hopf transformation.
    
    PDE: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    IC:  u(x,0) = -sin(π(x + 0.5)) on x ∈ [-0.5, 1.0]
    
    Solution:
        u(x,t) = -2ν·∂φ/∂x / φ
    where
        φ(x,t) = ∫_{-∞}^{∞} exp(-(x-ξ)²/(4νt) - F(ξ)/(2ν)) dξ
        F(ξ) = ∫₀^ξ u₀(s) ds = ∫₀^ξ -sin(π(s+0.5)) ds = [cos(π(s+0.5))/π]₀^ξ
             = (cos(π(ξ+0.5)) - cos(π·0.5)) / π
             = (cos(π(ξ+0.5)) - 0) / π  (if 0.5 shift)
    
    NOTE! Need to account that F(ξ) is the antiderivative of u₀:
        F'(ξ) = u₀(ξ) = -sin(π(ξ+0.5))
        F(ξ) = cos(π(ξ+0.5))/π + C
    """
    from scipy.integrate import quad
    import numpy as np
    import torch
    
    x, t = coords[:, 0:1].numpy(), coords[:, 1:2].numpy()
    
    # Antiderivative of IC:
    def F_primitive(xi):
        """∫ u₀(s) ds = ∫ -sin(π(s+0.5)) ds"""
        return np.cos(np.pi * (xi)) / np.pi
    
    result = np.zeros_like(x)
    
    for i in range(len(x)):
        xi = x[i, 0]
        ti = t[i, 0]
        
        # At t≈0 return IC:
        if ti < 1e-8:
            result[i] = -np.sin(np.pi * xi)
            continue
        
        # Adaptive integration limits:
        sigma = np.sqrt(4 * nu * ti)  # Standard deviation of gaussian
        limit_low = xi - 6 * sigma    # -6σ
        limit_high = xi + 6 * sigma   # +6σ
        
        # Limit by domain bounds (with margin):
        limit_low = max(limit_low, -5.0)
        limit_high = min(limit_high, 5.0)
        
        # Integrals for numerator and denominator:
        def integrand_phi(eta):
            """Integrand for φ"""
            gauss = np.exp(-(xi - eta)**2 / (4 * nu * ti))
            hopf = np.exp(-F_primitive(eta) / (2 * nu))
            return gauss * hopf
        
        def integrand_phi_x(eta):
            """Integrand for ∂φ/∂x"""
            gauss_deriv = -(xi - eta) / (2 * nu * ti) * np.exp(-(xi - eta)**2 / (4 * nu * ti))
            hopf = np.exp(-F_primitive(eta) / (2 * nu))
            return gauss_deriv * hopf
        
        try:
            # Compute φ and ∂φ/∂x:
            phi, _ = quad(integrand_phi, limit_low, limit_high, 
                         limit=100, epsabs=1e-10, epsrel=1e-8)
            phi_x, _ = quad(integrand_phi_x, limit_low, limit_high, 
                           limit=100, epsabs=1e-10, epsrel=1e-8)
            
            # Cole-Hopf formula:
            if abs(phi) > 1e-12:
                result[i] = -2 * nu * phi_x / phi
            else:
                # Fallback for small phi:
                result[i] = -np.sin(np.pi * (xi))
        
        except Exception as e:
            print(f"Warning: Integration failed at x={xi:.3f}, t={ti:.3f}: {e}")
            result[i] = -np.sin(np.pi * (xi))
    
    return torch.tensor(result, dtype=coords.dtype, device=coords.device)


def get_exact_solution_heat(alpha=1.0):
    """Exact solution for heat equation with parameter alpha"""
    def solution(coords):
        x, t = coords[:, 0:1], coords[:, 1:2]
        return torch.sin(np.pi * x) * torch.exp(-alpha * np.pi**2 * t)
    return solution


def get_exact_solution_wave(c=1.0):
    """Exact solution for wave equation with parameter c"""
    def solution(coords):
        x, t = coords[:, 0:1], coords[:, 1:2]
        return torch.sin(np.pi * x) * torch.cos(c * np.pi * t)
    return solution


def get_exact_solution_burgers(nu=0.01):
    """Exact solution for Burgers equation with parameter nu"""
    def solution(coords):
        return compute_burgers_exact(coords, nu)
    return solution


def get_exact_solution_reaction_diffusion(D=0.01, r=1.0):
    """
    Exact solution for reaction-diffusion (traveling wave).
    Wave speed: v = sqrt(D * r / 2)
    """
    def solution(coords):
        x, t = coords[:, 0:1], coords[:, 1:2]
        v = np.sqrt(D * r / 2)  # front speed
        return 0.5 * (1 + torch.tanh((x - v * t) / np.sqrt(2 * D)))
    return solution


def get_exact_solution_parametric(pde_type: str, **params):
    """
    Universal function to get exact solution with parameters.
    
    Args:
        pde_type: PDE type
        **params: PDE parameters (alpha, c, nu, D, r)
    
    Returns:
        exact solution function
    """
    if pde_type == 'heat':
        alpha = params.get('alpha', 1.0)
        return get_exact_solution_heat(alpha)
    
    elif pde_type == 'wave':
        c = params.get('c', 1.0)
        return get_exact_solution_wave(c)
    
    elif pde_type == 'burgers':
        nu = params.get('nu', 0.01)
        return get_exact_solution_burgers(nu)
    
    elif pde_type == 'reaction_diffusion':
        D = params.get('D', 0.01)
        r = params.get('r', 1.0)
        return get_exact_solution_reaction_diffusion(D, r)
    
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")