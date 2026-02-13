import torch
import numpy as np
from utils.pde_utils import get_exact_solution_parametric
from utils.config import DOMAINS

def generate_synthetic_data(pde_type: str, domain: dict, n_points: int = 100, 
                            noise_level: float = 0.0, sampling: str = 'random',
                            seed: int = None, true_params: dict = None):
    """
    Генерация псевдо-экспериментальных данных из точного решения + шум.
    
    Args:
        pde_type: тип PDE ('heat', 'wave', 'burgers', 'reaction_diffusion')
        domain: словарь с границами {'x': (min, max), 't': (min, max)}
        n_points: количество точек данных
        noise_level: уровень шума (0.0 = без шума, 0.05 = 5% от std данных)
        sampling: стратегия сэмплирования
        seed: random seed для воспроизводимости
        true_params: истинные параметры PDE для генерации данных
                     Например: {'alpha': 1.5} для heat, {'nu': 0.02} для burgers
    
    Returns:
        data_points: torch.Tensor [n_points, 2] — координаты (x, t)
        data_values: torch.Tensor [n_points, 1] — зашумлённые значения u
        data_clean: torch.Tensor [n_points, 1] — чистые значения (для анализа)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x_min, x_max = domain['x']
    t_min, t_max = domain['t']
    
    # Генерация точек в зависимости от стратегии
    if sampling == 'random':
        x = torch.rand(n_points, 1) * (x_max - x_min) + x_min
        t = torch.rand(n_points, 1) * (t_max - t_min) + t_min
        
    elif sampling == 'grid':
        n_x = int(np.sqrt(n_points))
        n_t = n_points // n_x
        x_lin = torch.linspace(x_min, x_max, n_x)
        t_lin = torch.linspace(t_min, t_max, n_t)
        X, T = torch.meshgrid(x_lin, t_lin, indexing='ij')
        x = X.flatten().unsqueeze(1)
        t = T.flatten().unsqueeze(1)
        
    elif sampling == 'sparse_time':
        n_time_slices = min(5, max(3, n_points // 20))
        t_slices = torch.linspace(t_min + 0.1*(t_max-t_min), 
                                   t_max - 0.1*(t_max-t_min), 
                                   n_time_slices)
        points_per_slice = n_points // n_time_slices
        
        x_list, t_list = [], []
        for t_val in t_slices:
            x_slice = torch.rand(points_per_slice, 1) * (x_max - x_min) + x_min
            t_slice = torch.full((points_per_slice, 1), t_val.item())
            x_list.append(x_slice)
            t_list.append(t_slice)
        
        x = torch.cat(x_list, dim=0)
        t = torch.cat(t_list, dim=0)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    
    data_points = torch.cat([x, t], dim=1)
    
    # Вычисление точного решения с параметрами
    from utils.pde_utils import get_exact_solution_parametric
    
    if true_params is not None:
        exact_solution = get_exact_solution_parametric(pde_type, **true_params)
    else:
        exact_solution = get_exact_solution_parametric(pde_type)  # defaults
    
    with torch.no_grad():
        data_clean = exact_solution(data_points)
        if data_clean.dim() == 1:
            data_clean = data_clean.unsqueeze(1)
    
    # Добавление шума
    if noise_level > 0:
        std_data = torch.std(data_clean)
        noise = torch.randn_like(data_clean) * noise_level * std_data
        data_values = data_clean + noise
    else:
        data_values = data_clean.clone()
    
    return data_points, data_values, data_clean


def generate_data_for_inverse(pde_type: str, domain: dict, n_points: int = 200,
                               noise_level: float = 0.05, seed: int = 42):
    """
    Удобная обёртка для inverse problem — больше точек, sparse sampling.
    """
    return generate_synthetic_data(
        pde_type=pde_type,
        domain=domain,
        n_points=n_points,
        noise_level=noise_level,
        sampling='sparse_time',
        seed=seed
    )


def get_default_domain(pde_type: str) -> dict:
    """Возвращает стандартный домен для каждого PDE (из твоих тестов)"""
    return DOMAINS[pde_type]