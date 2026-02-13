DOMAINS = {
    'heat': {'x': (0.0, 1.0), 't': (0.0, 0.5)},
    'wave': {'x': (0.0, 1.0), 't': (0.0, 1.0)},
    'burgers': {'x': (-1.0, 1.0), 't': (0.0, 0.5)},
    'reaction_diffusion': {'x': (-1.0, 1.0), 't': (0.0, 1.0)}
}

TITLES = {
    'heat': 'Heat Equation',
    'wave': 'Wave Equation',
    'burgers': 'Burgers Equation',
    'reaction_diffusion': 'Reaction-Diffusion'
}

PDE_CONFIGS = {
    'heat': {'n_steps': 4, 'n_iterations': 2, 'lr': 0.005, 'theta':0.4},
    'wave': {'n_steps': 3, 'n_iterations': 3, 'lr': 0.001, 'theta':0.5},
    'burgers': {'n_steps': 2, 'n_iterations': 2, 'lr': 0.0005, 'theta':0.3},
    'reaction_diffusion': {'n_steps': 2, 'n_iterations': 2, 'lr': 0.005, 'theta':0.3},
}

FEATURE_SETS = {
    'minimal': ['x', 't'],  
    'no_grad': ['x', 't', 't_next'], 
    'default': ['x', 't', 't_next', 'grad_norm'],
    'with_h': ['x', 't', 'h', 'grad_norm'], 
    'with_t_norm': ['x', 't_norm', 'grad_norm'],
    'with_y': ['x', 't', 'grad_norm', 'y'],
    'with_y_abs': ['x', 't', 'grad_norm', 'y_abs'],
    'full': ['x', 't', 't_next', 'h', 'grad_norm', 'y'],
    'with_laplacian': ['x', 't', 'grad_norm', 'laplacian'],
    'all': ['x', 't', 't_next', 'h', 't_norm', 'grad_norm', 'y', 'y_abs', 'laplacian'],
}