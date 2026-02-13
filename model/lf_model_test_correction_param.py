import torch
import torch.nn as nn
import numpy as np


class LFPinn_ThetaParams(nn.Module):
    """
    LF-PINN с лёгкими обучаемыми параметрами для коррекции.
    
    Режимы коррекции (correction_mode):
    
        --- Theta-коррекции по индексу шага ---
        
        'none'            — Без поправок (baseline, finetune theta_net на данных)
        
        'per_step_bias'   — θ_i = clamp(θ_base + δ_i, 0, 1)
                            Параметров: n_steps
        
        'per_step_affine' — θ_i = clamp(a_i · θ_base + b_i, 0, 1)
                            Параметров: 2 * n_steps
        
        'per_step_gate'   — θ_i = (1 - σ(g_i)) · θ_base + σ(g_i) · σ(μ_i)
                            Параметров: 2 * n_steps
        
        'shared_bias'     — θ = clamp(θ_base + δ, 0, 1)
                            Параметров: 1
        
        --- Theta-коррекции по реальному времени ---
        
        'time_bias'       — δ = interp(anchors, t_mid / t_max)
                            θ = clamp(θ_base + δ, 0, 1)
                            Параметров: n_time_anchors
        
        'time_gate'       — g, μ = interp(anchors, t_mid / t_max)
                            θ = (1 - σ(g)) · θ_base + σ(g) · σ(μ)
                            Параметров: 2 * n_time_anchors
        
        --- Не-theta коррекции ---
        
        'rhs_scale'       — y_new = y + exp(ε_i) · h · update
                            Параметров: n_steps
        
        'output_bias'     — y = y_new + b_i
                            Параметров: n_steps
        
        --- Комбинированные ---
        
        'step_bias+output_bias'  — per_step_bias (θ) + output_bias (y)
                                   Параметров: 2 * n_steps
        
        'time_bias+output_bias'  — time_bias (θ) + output_bias (y)
                                   Параметров: n_time_anchors + n_steps
        
        'time_gate+rhs_scale'    — time_gate (θ) + rhs_scale (update)
                                   Параметров: 2 * n_time_anchors + n_steps
    """
    
    AVAILABLE_MODES = [
        'none', 'per_step_bias', 'per_step_affine', 'per_step_gate', 'shared_bias',
        'time_bias', 'time_gate',
        'rhs_scale', 'output_bias',
        'step_bias+output_bias', 'time_bias+output_bias', 'time_gate+rhs_scale',
    ]
    
    # Какие комбо-режимы используют time anchors
    _TIME_MODES = ('time_bias', 'time_gate', 'time_bias+output_bias', 'time_gate+rhs_scale')
    # Какие комбо-режимы включают output_bias
    _OUTPUT_BIAS_MODES = ('output_bias', 'step_bias+output_bias', 'time_bias+output_bias')
    # Какие комбо-режимы включают rhs_scale
    _RHS_SCALE_MODES = ('rhs_scale', 'time_gate+rhs_scale')
    
    def __init__(self, pde_type: str,
                 correction_mode: str = 'none',
                 n_steps: int = 5,
                 theta_hidden_dim: int = 2,
                 n_iterations: int = 2,
                 initial_theta: float = 0.5,
                 lr: float = 0.001,
                 n_time_anchors: int = 5,
                 t_max: float = 1.0):
        super().__init__()
        
        self.pde_type = pde_type
        self.n_steps = n_steps
        self.is_wave = (pde_type == 'wave')
        self.n_iterations = n_iterations
        self.lr = lr
        self.initial_theta = initial_theta
        self.n_time_anchors = n_time_anchors
        self.t_max = t_max
        
        if correction_mode not in self.AVAILABLE_MODES:
            raise ValueError(f"Unknown mode: {correction_mode}. Available: {self.AVAILABLE_MODES}")
        self.correction_mode = correction_mode
        
        # BASE THETA NETWORK
        self.theta_net = nn.Sequential(
            nn.Linear(4, theta_hidden_dim), nn.Tanh(),
            nn.Linear(theta_hidden_dim, 1), nn.Sigmoid()
        )
        
        # CORRECTION PARAMETERS
        self._init_correction_params(correction_mode, n_steps)
        
        # Якоря по времени
        if correction_mode in self._TIME_MODES:
            self.register_buffer(
                'time_anchors',
                torch.linspace(0.0, 1.0, n_time_anchors)
            )
        
        self.use_corrections = False
        
        self._init_pde_params()
        self._init_theta_bias(initial_theta)
        self._print_info()
    
    def _init_correction_params(self, mode, n_steps):
        if mode == 'per_step_bias':
            self.step_bias = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'per_step_affine':
            self.step_scale = nn.Parameter(torch.ones(n_steps))
            self.step_shift = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'per_step_gate':
            self.gate_logit = nn.Parameter(torch.full((n_steps,), -3.0))
            self.target_logit = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'shared_bias':
            self.shared_bias = nn.Parameter(torch.tensor(0.0))
        elif mode == 'time_bias':
            self.time_bias_values = nn.Parameter(torch.zeros(self.n_time_anchors))
        elif mode == 'time_gate':
            self.time_gate_logit = nn.Parameter(torch.full((self.n_time_anchors,), -3.0))
            self.time_target_logit = nn.Parameter(torch.zeros(self.n_time_anchors))
        elif mode == 'rhs_scale':
            self.rhs_log_scale = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'output_bias':
            self.step_output_bias = nn.Parameter(torch.zeros(n_steps))
        # --- Комбинированные ---
        elif mode == 'step_bias+output_bias':
            self.step_bias = nn.Parameter(torch.zeros(n_steps))
            self.step_output_bias = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'time_bias+output_bias':
            self.time_bias_values = nn.Parameter(torch.zeros(self.n_time_anchors))
            self.step_output_bias = nn.Parameter(torch.zeros(n_steps))
        elif mode == 'time_gate+rhs_scale':
            self.time_gate_logit = nn.Parameter(torch.full((self.n_time_anchors,), -3.0))
            self.time_target_logit = nn.Parameter(torch.zeros(self.n_time_anchors))
            self.rhs_log_scale = nn.Parameter(torch.zeros(n_steps))
    
    def _init_theta_bias(self, target_theta: float):
        with torch.no_grad():
            target_clamped = np.clip(target_theta, 0.01, 0.99)
            bias_value = np.log(target_clamped / (1 - target_clamped))
            nn.init.xavier_normal_(self.theta_net[-2].weight, gain=0.1)
            self.theta_net[-2].bias.fill_(bias_value)
    
    def _init_pde_params(self):
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
        n_base = sum(p.numel() for p in self.theta_net.parameters())
        n_corr = self._count_correction_params()
        print(f"\n{'='*60}")
        print(f"LF-PINN Theta Params")
        print(f"PDE: {self.pde_type} | Mode: {self.correction_mode}")
        print(f"Steps: {self.n_steps} | Iterations: {self.n_iterations}")
        if self.correction_mode in self._TIME_MODES:
            print(f"Time anchors: {self.n_time_anchors} | t_max: {self.t_max}")
        print(f"Base theta_net params: {n_base}")
        print(f"Correction params: {n_corr}")
        print(f"Total trainable: {n_base + n_corr}")
        print(f"{'='*60}\n")
    
    def _count_correction_params(self):
        counts = {
            'none': 0,
            'per_step_bias': self.n_steps,
            'per_step_affine': 2 * self.n_steps,
            'per_step_gate': 2 * self.n_steps,
            'shared_bias': 1,
            'time_bias': self.n_time_anchors,
            'time_gate': 2 * self.n_time_anchors,
            'rhs_scale': self.n_steps,
            'output_bias': self.n_steps,
            'step_bias+output_bias': 2 * self.n_steps,
            'time_bias+output_bias': self.n_time_anchors + self.n_steps,
            'time_gate+rhs_scale': 2 * self.n_time_anchors + self.n_steps,
        }
        return counts.get(self.correction_mode, 0)
    
    # ============================================================
    # CORRECTION CONTROL
    # ============================================================
    
    def get_correction_param_names(self):
        names = {
            'per_step_bias': ['step_bias'],
            'per_step_affine': ['step_scale', 'step_shift'],
            'per_step_gate': ['gate_logit', 'target_logit'],
            'shared_bias': ['shared_bias'],
            'time_bias': ['time_bias_values'],
            'time_gate': ['time_gate_logit', 'time_target_logit'],
            'rhs_scale': ['rhs_log_scale'],
            'output_bias': ['step_output_bias'],
            'step_bias+output_bias': ['step_bias', 'step_output_bias'],
            'time_bias+output_bias': ['time_bias_values', 'step_output_bias'],
            'time_gate+rhs_scale': ['time_gate_logit', 'time_target_logit', 'rhs_log_scale'],
            'none': []
        }
        return names.get(self.correction_mode, [])
    
    def get_correction_params(self):
        return [getattr(self, name) for name in self.get_correction_param_names()]
    
    def get_base_params(self):
        return list(self.theta_net.parameters())
    
    def enable_corrections(self):
        self.use_corrections = True
        for name in self.get_correction_param_names():
            getattr(self, name).requires_grad_(True)
        print(f"[ThetaParams] Corrections ENABLED (mode: {self.correction_mode})")
    
    def disable_corrections(self):
        self.use_corrections = False
        print(f"[ThetaParams] Corrections DISABLED")
    
    def freeze_corrections(self):
        for name in self.get_correction_param_names():
            getattr(self, name).requires_grad_(False)
        print(f"[ThetaParams] Correction params FROZEN")
    
    def unfreeze_corrections(self):
        for name in self.get_correction_param_names():
            getattr(self, name).requires_grad_(True)
        print(f"[ThetaParams] Correction params UNFROZEN")
    
    def freeze_base_theta(self):
        for param in self.theta_net.parameters():
            param.requires_grad = False
        print("[ThetaParams] Base theta_net FROZEN")
    
    def unfreeze_base_theta(self):
        for param in self.theta_net.parameters():
            param.requires_grad = True
        print("[ThetaParams] Base theta_net UNFROZEN")
    
    def reset_corrections(self):
        with torch.no_grad():
            mode = self.correction_mode
            if mode == 'per_step_bias':
                self.step_bias.fill_(0.0)
            elif mode == 'per_step_affine':
                self.step_scale.fill_(1.0)
                self.step_shift.fill_(0.0)
            elif mode == 'per_step_gate':
                self.gate_logit.fill_(-3.0)
                self.target_logit.fill_(0.0)
            elif mode == 'shared_bias':
                self.shared_bias.fill_(0.0)
            elif mode == 'time_bias':
                self.time_bias_values.fill_(0.0)
            elif mode == 'time_gate':
                self.time_gate_logit.fill_(-3.0)
                self.time_target_logit.fill_(0.0)
            elif mode == 'rhs_scale':
                self.rhs_log_scale.fill_(0.0)
            elif mode == 'output_bias':
                self.step_output_bias.fill_(0.0)
            elif mode == 'step_bias+output_bias':
                self.step_bias.fill_(0.0)
                self.step_output_bias.fill_(0.0)
            elif mode == 'time_bias+output_bias':
                self.time_bias_values.fill_(0.0)
                self.step_output_bias.fill_(0.0)
            elif mode == 'time_gate+rhs_scale':
                self.time_gate_logit.fill_(-3.0)
                self.time_target_logit.fill_(0.0)
                self.rhs_log_scale.fill_(0.0)
        print("[ThetaParams] Corrections RESET")
    
    def change_n_steps(self, new_n_steps: int):
        """Изменить число шагов и пересоздать step-зависимые correction params."""
        old_n_steps = self.n_steps
        self.n_steps = new_n_steps
        
        # Чисто time-based — ничего не меняем
        if self.correction_mode in ('time_bias', 'time_gate'):
            print(f"[ThetaParams] n_steps: {old_n_steps} → {new_n_steps} "
                  f"(time-based corrections unchanged)")
            return
        
        # Комбинированные с time: пересоздаём только step-часть
        if self.correction_mode in ('time_gate+rhs_scale', 'time_bias+output_bias'):
            step_params = {'time_gate+rhs_scale': 'rhs_log_scale',
                           'time_bias+output_bias': 'step_output_bias'}
            name = step_params[self.correction_mode]
            if hasattr(self, name):
                delattr(self, name)
            setattr(self, name, nn.Parameter(torch.zeros(new_n_steps)))
            print(f"[ThetaParams] n_steps: {old_n_steps} → {new_n_steps} "
                  f"(time corrections unchanged, {name} reinit)")
            return
        
        # Всё остальное — пересоздаём полностью
        for name in self.get_correction_param_names():
            if hasattr(self, name):
                delattr(self, name)
        self._init_correction_params(self.correction_mode, new_n_steps)
        
        print(f"[ThetaParams] n_steps: {old_n_steps} → {new_n_steps} "
              f"(correction params reinit, {self._count_correction_params()} params)")
    
    # ============================================================
    # TIME INTERPOLATION
    # ============================================================
    
    def _interp_time(self, values: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        """Линейная интерполяция значений по нормализованному времени [0, 1]."""
        t_frac = t_frac.clamp(0.0, 1.0)
        n = len(self.time_anchors)
        
        idx_float = t_frac * (n - 1)
        idx_low = idx_float.long().clamp(0, n - 2)
        idx_high = idx_low + 1
        w = idx_float - idx_low.float()
        
        val_low = values[idx_low.squeeze(-1)]
        val_high = values[idx_high.squeeze(-1)]
        
        result = (1.0 - w.squeeze(-1)) * val_low + w.squeeze(-1) * val_high
        return result.unsqueeze(-1)
    
    def _get_t_frac(self, inputs: torch.Tensor) -> torch.Tensor:
        """Нормализованное время середины шага."""
        t_mid = (inputs[:, 1:2] + inputs[:, 2:3]) / 2.0
        return t_mid / self.t_max
    
    # ============================================================
    # THETA COMPUTATION
    # ============================================================
    
    def get_theta(self, inputs: torch.Tensor, step: int) -> torch.Tensor:
        """Вычислить θ с учётом коррекции. inputs: [batch, 4] — (x, t, t_next, grad_norm)."""
        theta_base = self.theta_net(inputs)
        
        if not self.use_corrections:
            return theta_base
        
        mode = self.correction_mode
        
        # Режимы, не трогающие theta
        if mode in ('none', 'rhs_scale', 'output_bias'):
            return theta_base
        
        # --- Step-index ---
        if mode == 'per_step_bias':
            return torch.clamp(theta_base + self.step_bias[step], 0.0, 1.0)
        
        if mode == 'per_step_affine':
            return torch.clamp(self.step_scale[step] * theta_base + self.step_shift[step], 0.0, 1.0)
        
        if mode == 'per_step_gate':
            gate = torch.sigmoid(self.gate_logit[step])
            target = torch.sigmoid(self.target_logit[step])
            return torch.clamp((1.0 - gate) * theta_base + gate * target, 0.0, 1.0)
        
        if mode == 'shared_bias':
            return torch.clamp(theta_base + self.shared_bias, 0.0, 1.0)
        
        # --- Time-based ---
        if mode in ('time_bias', 'time_bias+output_bias'):
            t_frac = self._get_t_frac(inputs)
            delta = self._interp_time(self.time_bias_values, t_frac)
            return torch.clamp(theta_base + delta, 0.0, 1.0)
        
        if mode in ('time_gate', 'time_gate+rhs_scale'):
            t_frac = self._get_t_frac(inputs)
            gate = torch.sigmoid(self._interp_time(self.time_gate_logit, t_frac))
            target = torch.sigmoid(self._interp_time(self.time_target_logit, t_frac))
            return torch.clamp((1.0 - gate) * theta_base + gate * target, 0.0, 1.0)
        
        # --- Комбо: step_bias + output_bias (theta часть) ---
        if mode == 'step_bias+output_bias':
            return torch.clamp(theta_base + self.step_bias[step], 0.0, 1.0)
        
        return theta_base
    
    # ============================================================
    # PDE FORWARD
    # ============================================================
    
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
        
        h = t_end / self.n_steps
        y = self.initial_condition(x)
        t = torch.zeros(batch_size, 1, dtype=dtype, device=device)
        
        for step in range(self.n_steps):
            t_next = t + h
            y_for_theta = y[:, 0:1] if self.is_wave else y
            
            y_x = torch.autograd.grad(
                y_for_theta.sum(), x,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if y_x is None:
                y_x = torch.zeros_like(x)
            grad_norm = torch.tanh(torch.abs(y_x) / 5.0)
            
            inputs = torch.cat([
                x.detach(), t.detach(), t_next.detach(), grad_norm.detach()
            ], dim=1)
            
            theta = self.get_theta(inputs, step)
            f_curr = self.compute_rhs(x, t, y)
            y_new = y.clone()
            
            for _ in range(self.n_iterations):
                f_next = self.compute_rhs(x, t_next, y_new)
                update = h * ((1 - theta) * f_curr + theta * f_next)
                
                # rhs_scale
                if self.use_corrections and self.correction_mode in self._RHS_SCALE_MODES:
                    scale = torch.exp(self.rhs_log_scale[step])
                    update = scale * update
                
                y_new = y + update
            
            # output_bias
            if self.use_corrections and self.correction_mode in self._OUTPUT_BIAS_MODES:
                bias = self.step_output_bias[step]
                if self.is_wave:
                    y_new = y_new + torch.cat([
                        bias.unsqueeze(0).expand(batch_size, 1),
                        torch.zeros(batch_size, 1, device=device)
                    ], dim=1)
                else:
                    y_new = y_new + bias
            
            y = y_new
            t = t_next
        
        return y[:, 0:1] if self.is_wave else y
    
    # ============================================================
    # LOSSES
    # ============================================================
    
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
        t_zero = torch.zeros_like(x_ic)
        u_pred = self.forward(x_ic, t_zero)
        u_true = self.initial_condition(x_ic)
        if self.is_wave:
            u_true = u_true[:, 0:1]
        return torch.mean((u_pred - u_true)**2)
    
    def total_loss(self, domain, n_collocation=30,
                   lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
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
    
    def data_loss(self, x_data, t_data, u_observed):
        u_pred = self.forward(x_data, t_data)
        return torch.mean((u_pred - u_observed)**2)
    
    # ============================================================
    # INFO & STATISTICS
    # ============================================================
    
    def get_params(self):
        return {
            'n_steps': self.n_steps,
            'n_iterations': self.n_iterations,
            'lr': self.lr,
            'correction_mode': self.correction_mode,
            'use_corrections': self.use_corrections,
            'initial_theta': self.initial_theta,
            'n_base_params': sum(p.numel() for p in self.theta_net.parameters()),
            'n_correction_params': self._count_correction_params(),
        }
    
    def get_theta_statistics(self, domain, n_samples=100):
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_sample = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
        t_sample = torch.rand(n_samples, 1, device=device) * (t_max - t_min) + t_min
        h = (t_max - t_min) / self.n_steps
        t_next = torch.clamp(t_sample + h, max=t_max)
        grad_norm = torch.zeros_like(x_sample)
        
        inputs = torch.cat([x_sample, t_sample, t_next, grad_norm], dim=1)
        
        with torch.no_grad():
            theta_values = self.get_theta(inputs, step=self.n_steps // 2)
        
        return {
            'mean': theta_values.mean().item(),
            'std': theta_values.std().item(),
            'min': theta_values.min().item(),
            'max': theta_values.max().item()
        }
    
    def get_correction_statistics(self, domain, n_samples=100):
        """Статистика коррекций по шагам."""
        if self.correction_mode == 'none':
            return None
        
        x_min, x_max = domain['x']
        t_min, t_max = domain['t']
        device = next(self.parameters()).device
        
        x_sample = torch.rand(n_samples, 1, device=device) * (x_max - x_min) + x_min
        t_sample = torch.rand(n_samples, 1, device=device) * (t_max - t_min) + t_min
        h = (t_max - t_min) / self.n_steps
        t_next = torch.clamp(t_sample + h, max=t_max)
        grad_norm = torch.zeros_like(x_sample)
        
        inputs = torch.cat([x_sample, t_sample, t_next, grad_norm], dim=1)
        
        stats = {}
        with torch.no_grad():
            theta_base = self.theta_net(inputs)
            
            for step in range(self.n_steps):
                key = f'step_{step}'
                theta_corrected = self.get_theta(inputs, step)
                correction = theta_corrected - theta_base
                
                stats[key] = {
                    'theta_base_mean': theta_base.mean().item(),
                    'theta_corrected_mean': theta_corrected.mean().item(),
                    'correction_mean': correction.mean().item(),
                    'correction_std': correction.std().item(),
                }
                
                mode = self.correction_mode
                if mode in ('per_step_bias', 'step_bias+output_bias'):
                    stats[key]['bias'] = self.step_bias[step].item()
                if mode == 'per_step_affine':
                    stats[key]['scale'] = self.step_scale[step].item()
                    stats[key]['shift'] = self.step_shift[step].item()
                if mode == 'per_step_gate':
                    stats[key]['gate'] = torch.sigmoid(self.gate_logit[step]).item()
                    stats[key]['target'] = torch.sigmoid(self.target_logit[step]).item()
                if mode == 'shared_bias':
                    stats[key]['shared_bias'] = self.shared_bias.item()
                if mode in self._RHS_SCALE_MODES:
                    stats[key]['rhs_scale'] = torch.exp(self.rhs_log_scale[step]).item()
                if mode in self._OUTPUT_BIAS_MODES:
                    stats[key]['output_bias'] = self.step_output_bias[step].item()
            
            # Time-based якоря
            if mode in ('time_bias', 'time_bias+output_bias'):
                for i in range(self.n_time_anchors):
                    t_anchor = self.time_anchors[i].item()
                    stats[f'anchor_{i}_t={t_anchor:.2f}'] = {
                        'bias': self.time_bias_values[i].item()
                    }
            elif mode in ('time_gate', 'time_gate+rhs_scale'):
                for i in range(self.n_time_anchors):
                    t_anchor = self.time_anchors[i].item()
                    stats[f'anchor_{i}_t={t_anchor:.2f}'] = {
                        'gate': torch.sigmoid(self.time_gate_logit[i]).item(),
                        'target': torch.sigmoid(self.time_target_logit[i]).item(),
                    }
        
        return stats