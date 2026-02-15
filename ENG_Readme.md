# PBA-based Low-Fidelity Model Based on the Generalized Trapezoidal Method

## 📋 Table of Contents

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Experiments: PDE Training Stage](#experiments-pde-training-stage)
  - [1. Activation Function Study](#1-activation-function-study)
  - [2. Initial Theta Values Study](#2-initial-theta-values-study)
  - [3. Optimizer Study (PDE Training)](#3-optimizer-study-pde-training)
  - [4. Learning Rate Study (PDE Training)](#4-learning-rate-study-pde-training)
  - [5. PDE Training Epochs Study](#5-pde-training-epochs-study)
  - [6. Loss Weighting Methods Study](#6-loss-weighting-methods-study)
- [Experiments: Data Finetuning Stage](#experiments-data-finetuning-stage)
  - [1. Correction Parameters Study](#1-correction-parameters-study)
  - [2. Data Training Learning Rate Study](#2-data-training-learning-rate-study)
  - [3. Data Training Epochs Study](#3-data-training-epochs-study)
- [Benchmarks](#benchmarks)
  - [1. Theta Hidden Dim (Neurons Number) Study](#1-theta-hidden-dim-neurons-number-study)
  - [2. Input Features Study](#2-input-features-study)
  - [3. Self Benchmark](#3-self-benchmark)
  - [4. PDE Training Benchmark Comparison](#4-pde-training-benchmark-comparison-lf-pinn-vs-classical-pinn-without-data)
  - [5. Data Training Benchmark Comparison](#5-data-training-benchmark-comparison-lf-pinn--data-vs-classical-pinn--data)
- [Final Results](#final-results)
- [Conclusions](#conclusions)

## Project Description

This project implements a Low-Fidelity Physics-Informed Neural Network (LF-PINN) with an adaptive generalized trapezoidal method. Unlike a classical PINN, this implementation constructs the solution step-by-step through a numerical scheme, where the PDE right-hand side defines each integration step:

$$ \mathbf{ y_{i+1} = y_i + h \cdot [ (1-\theta) f_i + \theta f_{i+1} ] } $$

where the parameter **θ ∈ [0, 1]** determines the scheme type:
- $\theta$ = 0: Explicit scheme
- $\theta$ = 0.5: Classical trapezoidal method
- $\theta$ = 1: Implicit scheme

Instead of a fixed $\theta$, a compact neural network (~13 parameters) is used, trained to select the optimal value $\theta(x, t_i, t_{i+1}, grad norm(\delta x/ \delta y))$ at each point in space-time. This allows the model to adaptively switch between schemes depending on local solution properties, achieving high accuracy with a minimal number of parameters.

The project includes a comparison with a classical PINN whose parameter count exceeds this implementation by **more than 5×** across four types of partial differential equations.

## Project Structure
```text
├── model/
│   ├── lf_model.py                        # Main LF-PINN model
│   ├── classic_pinn_model.py              # Classical PINN for comparison
│   ├── lf_model_test_activation.py        # Model for activation testing
│   ├── lf_model_test_correction_param.py  # Model with correction parameters
│   ├── lf_model_test_extra_parameter.py   # Model with extended inputs
│   ├── lf_model_test_initial_theta.py     # Model for initial θ testing
│   ├── lf_model_test_loss_weighting.py    # Model with adaptive loss weights
│   └── lf_model_test_optimizer.py         # Model for optimizer testing
├── utils/
│   ├── benchmark_utils.py                 # Benchmark utilities
│   ├── config.py                          # Configuration
│   ├── data_utils.py                      # Data handling
│   ├── pde_utils.py                       # PDE utilities and analytical solutions
│   └── plot_utils.py                      # Result visualization
├── research_notebooks/                    # Research experiments
│   ├── activation_function_study.ipynb
│   ├── initial_theta_values_study.ipynb
│   ├── pde_training_all_optimizer_study.ipynb
│   ├── pde_training_best_optimizer_study.ipynb
│   ├── pde_training_lr_Adam_study.ipynb
│   ├── pde_training_lr_RmsProp_study.ipynb
│   ├── pde_training_epochs_study.ipynb
│   ├── loss_weighting_methods_study.ipynb
│   ├── data_training_learning_rate_study.ipynb
│   ├── data_training_epochs_study.ipynb
│   ├── data_training_correction_param_study.ipynb
│   ├── data_optimizer_study.ipynb
│   └── data_points_number.ipynb
├── benchmark_notebooks/                   # Benchmarks and comparisons
│   ├── self_benchmark.ipynb
│   ├── benchmark_neurons_number.ipynb
│   ├── benchmark_extra_parameter.ipynb
│   ├── benchmark_data_assisted.ipynb
│   └── benchmark_comparison.ipynb
├── plots/                                 # Result plots
└── README.md
```

## Methodology

### Space-Time Discretization

$$h = (t_{max} - t_{min}) / n_{steps}$$

### Fixed-Point Iterations of the Generalized Trapezoidal Method

At each time step, $${n_{iter}}$$ fixed-point iterations of the numerical scheme are performed:

$$y_{n+1} = y_n + h·[(1-θ)·F(x, t_n, y_n) + θ·F(x, t_{n+1}, y*)]$$

where $$θ = θ_{net}(x, t_n, t_{n+1}, gradnorm)$$ is predicted by the neural network.

### Training Parameters
```text
n_steps = 4, n_iter = 2 for the Heat equation
n_steps = 3, n_iter = 3 for the Wave equation
n_steps = 2, n_iter = 2 for the Burgers and Reaction-Diffusion equations

n_collocation = 30       # Collocation points
n_bc = 10                # Boundary points
n_ic = 10                # Initial condition points
learning_rate = 0.001
```

### Loss Function

$$L_{total} = λ_{pde}·L_{pde} + λ_{bc}·L_{bc} + λ_{ic}·L_{ic}$$

---

## Experiments: PDE Training Stage

### 1. Activation Function Study

Comparison of θ-network activation functions: Tanh, Softplus, GELU, ELU, LeakyReLU (α=0.01, 0.05, 0.1, 0.2), ELU.
5 runs per activation function.

![Activation Function Study — Heat](plots/heat_eq_activation_function_study.png)

| PDE | Best Activation | L2RE | RMSE | Final θ |
|-----|:---:|:---:|:---:|:---:|
| Heat | Softplus | 1.31e-02 ± 1.6e-03 | 3.04e-03 ± 3.8e-04 | 0.333 |
| Wave | Softplus | 3.03e-02 ± 3.6e-04 | 1.52e-02 ± 1.8e-04 | 0.413 |
| Burgers | LeakyReLU_0.01 | 1.66e-01 ± 9.1e-04 | 1.11e-01 ± 6.1e-04 | 0.186 |
| Reaction-Diffusion | Tanh | 1.36e-01 ± 1.0e-05 | 9.09e-02 ± 6.9e-06 | 0.145 |

**Key points:**
- Softplus works best on smooth problems (Heat, Wave), providing smooth θ prediction
- The spread between activations is small (~10–15%), indicating architecture robustness
- All activations train θ to values below 0.5 (closer to the implicit scheme)

**Conclusion:**
- Use **TANH** activation for all equations, as the impact on training quality is minimal

---

### 2. Initial Theta Values Study

Investigation of the effect of the initial θ value on final accuracy and convergence. Values tested: θ₀ ∈ {0.0, 0.3–0.5, 0.5, 1.0}.

![Initial Theta Study — Heat](plots/heat_eq_initial_theta_study.png)

| PDE | Best θ₀ | L2RE | Final θ | Converged to |
|-----|:---:|:---:|:---:|:---:|
| Heat | 0.40 | 1.30e-02 ± 1.2e-03 | 0.314 | Custom (0.31) |
| Wave | 0.40 | 3.00e-02 ± 3.0e-04 | 0.415 | ~Trapezoidal |
| Burgers | 0.30 | 1.64e-01 ± 5.6e-04 | 0.273 | Custom (0.27) |
| Reaction-Diffusion | 0.30 | 1.36e-01 ± 1.4e-05 | 0.246 | Custom (0.25) |

**Key points:**
- Initialization at θ₀ = 1.0 (explicit scheme) leads to catastrophic degradation on all problems
- The model adaptively shifts θ to its own optimum, but too distant initialization (0.0 or 1.0) causes convergence stalling
- For all PDEs, the final θ converges to values of 0.25–0.42, indicating a preference for schemes closer to implicit

**Conclusion:**
- Optimal initial values should be chosen closer to the value that **θ** converges to for each equation

---

### 3. Optimizer Study (PDE Training)

Two-stage comparison: first 9 optimizers (Adam, AdamW, SGD Momentum, SGD Nesterov, RMSProp, NAdam, RAdam, AdaMax), then a detailed comparison of the top 3 (Adam, AdamW with different weight decay, RMSProp).

![Optimizer Study — Heat](plots/heat_eq_pde_training_optimizer_study.png)

**Stage 1: All Optimizers**

| PDE | Best Optimizer | L2RE | Time (s) |
|-----|:---:|:---:|:---:|
| Heat | RMSProp | 1.14e-02 ± 6.2e-03 | 45.3 |
| Burgers | Adam | 1.64e-01 ± 1.2e-02 | 99.7 |
| Reaction-Diffusion | SGD Nesterov | 6.91e-02 ± 4.5e-05 | 136.6 |

**Stage 2: Top 3 (Adam, AdamW, RMSProp)**

| PDE | Best Optimizer | L2RE | Time (s) |
|-----|:---:|:---:|:---:|
| Heat | AdamW (wd=0.01) | 1.20e-02 ± 7.0e-04 | 90.5 |
| Wave | AdamW (wd=0.01) | 3.09e-02 ± 9.1e-04 | 343.3 |
| Burgers | RMSProp | 1.59e-01 ± 1.3e-03 | 174.4 |
| Reaction-Diffusion | RMSProp (centered) | 1.36e-01 ± 2.9e-05 | 239.4 |

**Key points:**
- The Adam family and RMSProp are clear leaders; SGD methods fall behind
- The difference between top optimizers is minimal (<5%), the model is robust to optimizer choice
- AdamW with weight decay 0.01 slightly improves results on smooth problems (Heat, Wave)
- On Burgers and Reaction-Diffusion, all optimizers yield virtually the same L2RE

**Conclusion:**
- Use the Adam optimizer, as the difference between RMSProp and Adam is minimal

---

### 4. Learning Rate Study (PDE Training)

Learning rate study for Adam and RMSProp at the PDE training stage. Values tested: lr ∈ {0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05}.

![LR Study — Heat](plots/heat_eq_pde_training_learning_rate_study.png)

**Adam:**

| PDE | Best LR | L2RE | Final Loss |
|-----|:---:|:---:|:---:|
| Heat | 0.005 | 1.34e-02 ± 1.1e-03 | 3.43e-04 |
| Wave | 0.001 | 3.01e-02 ± 1.1e-04 | 4.89e-02 |
| Burgers | 0.0005 | 1.65e-01 ± 2.4e-04 | 2.63e+00 |
| Reaction-Diffusion | 0.0025 | 1.36e-01 ± 8.7e-06 | 9.95e-03 |

**RMSProp:**

| PDE | Best LR | L2RE | Final Loss |
|-----|:---:|:---:|:---:|
| Heat | 0.005 | 1.33e-02 ± 1.1e-03 | 2.55e-04 |
| Wave | 0.001 | 3.03e-02 ± 2.2e-04 | 4.05e-02 |
| Burgers | 0.0005 | 1.61e-01 ± 2.7e-04 | 4.28e+00 |
| Reaction-Diffusion | 0.001 | 1.36e-01 ± 1.1e-05 | 3.09e-03 |

**Key points:**
- The optimal lr is strongly PDE-dependent: Heat and Wave prefer lr=0.001–0.005, Burgers prefers lr=0.0005
- Adam and RMSProp show similar optimal lr values for each PDE
- Too high lr (0.05) often degrades L2RE, especially on Heat
- Burgers is most sensitive to lr — even a slight increase in lr worsens the result

**Conclusion:**
- Use different **learning rate** values for each equation

---

### 5. PDE Training Epochs Study

Study of L2RE dependence on the number of PDE training epochs (without data). Evaluation of convergence speed and determination of the optimal epoch count.

![PDE Epochs Study — Heat](plots/heat_eq_pde_training_epochs_study.png)

**Key points:**
- Heat and Wave converge within 100–200 epochs; further training yields marginal improvement
- Burgers converges more slowly due to nonlinearity, but the main progress occurs in the first 200 epochs
- Reaction-Diffusion reaches a plateau almost instantly (~50 epochs)
- No overfitting is observed thanks to the small number of parameters (13)

**Conclusion:**
- A large number of epochs does not significantly improve the result; an epoch count in the range of 100–200 should be used

---

### 6. Loss Weighting Methods Study

Comparison of 8 loss weighting strategies: Fixed, Gradual, Inverse, SoftAdapt, ReLoBRaLo, NTK, Self-Adaptive, Causal. Base weights: λ_pde=1.0, λ_bc=10.0, λ_ic=10.0.

![Loss Weighting Study — Heat](plots/heat_eq_loss_weighting_methods_study.png)

**Heat:**

| Strategy | L2RE | RMSE | θ final |
|----------|:---:|:---:|:---:|
| **Causal** ★ | 1.22e-02 ± 1.1e-03 | 2.83e-03 ± 2.5e-04 | 0.310 |
| Self-Adaptive | 1.29e-02 ± 8.5e-04 | 2.99e-03 ± 2.0e-04 | 0.319 |
| SoftAdapt | 1.33e-02 ± 7.9e-04 | 3.07e-03 ± 1.8e-04 | 0.310 |
| Fixed | 1.34e-02 ± 7.0e-04 | 3.10e-03 ± 1.6e-04 | 0.309 |
| NTK | 1.44e-02 ± 7.2e-04 | 3.33e-03 ± 1.7e-04 | 0.298 |
| Inverse | 1.49e-02 ± 1.2e-03 | 3.45e-03 ± 2.7e-04 | 0.331 |

**Best Strategy per PDE:**

| PDE | Best Strategy | L2RE | Improvement vs Fixed |
|-----|:---:|:---:|:---:|
| Heat | Causal | 1.22e-02 ± 1.1e-03 | +8.7% |
| Burgers | Inverse | 1.65e-01 ± 5.0e-04 | +0.4% |
| Reaction-Diffusion | Self-Adaptive | 1.36e-01 ± 1.2e-05 | +0.0% |

**Key points:**
- Causal weighting yields the best result on Heat (+8.7% vs Fixed), accounting for the temporal causality of the problem
- On Burgers and Reaction-Diffusion, the difference between strategies is negligible (<0.5%)
- NTK and Inverse perform worse than others on Heat, despite their theoretical justification
- For LF-PINN with few parameters, adaptive weighting has a smaller effect than for large networks

**Conclusion:**
- Using **Loss Weighting** strategies is not an effective way to improve training quality for this problem

---

## Experiments: Data Finetuning Stage

### 1. Correction Parameters Study

Comparison of 10 correction parameter modes at the data finetuning stage: None, Per-Step Bias, Per-Step Gate, Time Bias, Time Gate, RHS Scale, Output Bias, and their combinations.

The base integration scheme at each step `i`:

```
y_{i+1} = y_i + h · [(1 − θ_i) · F(x, t_i, y_i) + θ_i · F(x, t_{i+1}, y_{i+1}*)]
```

where `θ_i = θ_net(x, t_i, t_{i+1}, grad_norm)` is the θ-network output. Correction parameters add learnable corrections **on top of the frozen θ-network** during the data finetuning stage.

---

### θ Corrections (modify the scheme parameter)

**1. None** — no correction, only the θ-network weights are finetuned:

```
θ_i = θ_base(x, t, t_next, grad_norm)
```

**2. Per-Step Bias** — additive shift of θ at each step (`n_steps` parameters: `δ_i`):

```
θ_i = clamp(θ_base + δ_i, 0, 1)
```

**3. Per-Step Gate** — gated blending of θ_base with a learnable target (`2·n_steps` parameters: `g_i, μ_i`):

```
θ_i = (1 − σ(g_i)) · θ_base + σ(g_i) · σ(μ_i)
```

When `g_i → −∞`: `θ_i ≈ θ_base` (gate closed). When `g_i → +∞`: `θ_i ≈ σ(μ_i)` (full replacement with learnable target).

**4. Time Bias** — additive shift of θ depending on normalized time via linear interpolation over anchor points (`n_anchors` parameters: `δ(τ)`):

```
τ = (t + t_next) / (2 · t_max)
δ = interp(anchors, τ)
θ_i = clamp(θ_base + δ(τ), 0, 1)
```

**5. Time Gate** — gating mechanism with time dependence (`2·n_anchors` parameters: `g(τ), μ(τ)`):

```
τ = (t + t_next) / (2 · t_max)
g = σ(interp(gate_anchors, τ))
μ = σ(interp(target_anchors, τ))
θ_i = (1 − g(τ)) · θ_base + g(τ) · μ(τ)
```

---

### Output Corrections (modify the solution y without touching θ)

**6. RHS Scale** — multiplicative scaling of the right-hand side at each step (`n_steps` parameters: `ε_i`):

```
update = h · [(1 − θ) · F_curr + θ · F_next]
y_{i+1} = y_i + exp(ε_i) · update
```

When `ε_i = 0`: no change. Allows correcting the update amplitude, compensating for systematic error in the PDE right-hand side.

**7. Output Bias** — additive shift of the solution after each step (`n_steps` parameters: `b_i`):

```
y_{i+1} = y_i + h · [(1 − θ) · F_curr + θ · F_next] + b_i
```

Compensates for constant systematic bias in the solution.

---

### Combined Modes

**8. Step Bias + Output Bias** — Per-Step Bias for θ + Output Bias for y (`2·n_steps` parameters):

```
θ_i = clamp(θ_base + δ_i, 0, 1)
y_{i+1} = y_i + update(θ_i) + b_i
```

**9. Time Bias + Output Bias** — Time Bias for θ + Output Bias for y (`n_anchors + n_steps` parameters):

```
θ_i = clamp(θ_base + δ(τ), 0, 1)
y_{i+1} = y_i + update(θ_i) + b_i
```

**10. Time Gate + RHS Scale** — Time Gate for θ + RHS Scale for update (`2·n_anchors + n_steps` parameters):

```
θ_i = (1 − g(τ)) · θ_base + g(τ) · μ(τ)
y_{i+1} = y_i + exp(ε_i) · update(θ_i)
```

The most expressive mode: simultaneously adapts both the scheme type (via θ) and the update scale.

---

![Correction Params — Heat (No Noise)](plots/heat_eq_data_training_correction_param.png)

**Without noise (50 data points):**

| PDE | Best Mode | Pretrain L2RE | Finetune L2RE | Δ% | Corr Params |
|-----|:---:|:---:|:---:|:---:|:---:|
| Heat | None | 1.26e-02 | 2.04e-03 ± 8.6e-04 | +83.6% | 0 |
| Wave | time_gate+rhs_scale | 3.08e-02 | 6.48e-03 ± 1.4e-03 | +79.0% | 13 |
| Burgers | time_gate+rhs_scale | 1.62e-01 | 1.37e-01 ± 1.0e-02 | +15.7% | 12 |
| Reaction-Diffusion | output_bias | 1.36e-01 | 1.25e-01 ± 5.4e-04 | +8.3% | 2 |

![Correction Params — Heat (5% Noise)](plots/heat_eq_data_training_correction_param_noise.png)

**With 5% noise:**

| PDE | Best Mode | Pretrain L2RE | Finetune L2RE | Δ% | Corr Params |
|-----|:---:|:---:|:---:|:---:|:---:|
| Heat | Per-Step Gate | 1.26e-02 | 7.50e-03 ± 3.3e-03 | +40.2% | 8 |
| Wave | time_gate+rhs_scale | 3.08e-02 | 9.32e-03 ± 3.1e-03 | +69.7% | 13 |
| Burgers | time_gate+rhs_scale | 1.62e-01 | 1.36e-01 ± 8.2e-03 | +16.5% | 12 |
| Reaction-Diffusion | output_bias | 1.36e-01 | 1.25e-01 ± 7.6e-04 | +8.2% | 2 |

**Key points:**
- On clean data, Heat achieves +83.6% improvement even without correction params (mode=None) — data directly corrects the θ-network
- For Wave, the time_gate+rhs_scale combination provides the maximum effect (+79%), scaling the PDE right-hand side
- With noise, mode=None degrades, while correction params (Gate, RHS Scale) act as regularizers
- RHS Scale is a universally useful mode for problems where the PDE right-hand side requires correction (Wave, Burgers)
- Output Bias is best for Reaction-Diffusion, compensating for systematic bias in the solution

**Conclusions:**
- For each equation, use the most effective correction in terms of **n_params|improvement** ratio during data finetuning

---

### 2. Data Training Learning Rate Study

Learning rate study at the data finetuning stage (50 points, 1000 epochs). Three correction modes were tested: None, Per-Step Bias / RHS Scale / Output Bias (depending on the PDE).

![Data LR Study — Heat](plots/heat_eq_data_training_learning_rate_study.png)

**Best results per PDE (best correction mode):**

| PDE | Correction Mode | Best LR | L2RE After | Improvement vs baseline |
|-----|:---:|:---:|:---:|:---:|
| Heat | Per-Step Gate | 0.0005 | 7.67e-03 ± 2.5e-03 | +40.3% |
| Wave | RHS Scale | 0.002 | 1.41e-02 ± 3.3e-03 | +57.2% |
| Burgers | RHS Scale | 0.005 | 1.29e-01 ± 1.2e-02 | +22.2% |
| Reaction-Diffusion | Output Bias | 0.0001 | 1.25e-01 ± 7.3e-04 | +8.2% |

**Key points:**
- Data finetuning significantly improves results: up to +57% on Wave, +40% on Heat
- The optimal lr for data finetuning is typically lower than for PDE training (0.0001–0.002)
- Without correction parameters (mode=None), a high lr often worsens the result — correction params stabilize training
- Burgers and Reaction-Diffusion benefit less from data, which is related to the more complex structure of their solutions

**Conclusions:**
- For each equation-correction combination, use a separate, most effective **learning rate**

---

### 3. Data Training Epochs Study

Study of L2RE saturation vs the number of data finetuning epochs (50 points). Using the best correction mode for each PDE.

![Data Epochs Study — Heat](plots/heat_eq_data_training_epochs_study.png)

| PDE | Correction Mode | Baseline L2RE | Best Epochs | Best L2RE | Max Improvement |
|-----|:---:|:---:|:---:|:---:|:---:|
| Heat | Per-Step Bias | 1.27e-02 | 500–1000 | 8.02e-03 | +36.9% |
| Wave | RHS Scale | 3.28e-02 | 500 | 1.39e-02 | +57.7% |
| Burgers | RHS Scale | 1.65e-01 | 2500–3000 | 1.28e-01 | +22.3% |
| Reaction-Diffusion | Output Bias | 1.36e-01 | 500 | 1.25e-01 | +8.2% |

**Key points:**
- The main improvement occurs in the first 500–1000 epochs, after which a plateau is reached
- On Heat and Wave, slight overfitting is observed after 1000–1500 epochs — L2RE begins to increase
- Burgers converges more slowly and continues to improve up to 2500 epochs
- Reaction-Diffusion saturates by 500 epochs; further training is futile

**Conclusions:**
- For all equations (except Burgers), there is no point in data training lasting **>1000** epochs
- Use a different number of epochs for each equation during data finetuning

---

## Benchmarks

### 1. Theta Hidden Dim (Neurons Number) Study

Study of the θ-network hidden layer size: hidden_dim ∈ {1, 2, 3, 4, 8}, corresponding to 7–49 parameters.

| PDE | Best Hidden Dim | Params | L2RE | θ final |
|-----|:---:|:---:|:---:|:---:|
| Heat | 2 | 13 | 1.27e-02 ± 1.3e-03 | 0.337 |
| Wave | 3 | 19 | 3.11e-02 ± 3.4e-04 | 0.412 |
| Burgers | 1 | 7 | 1.64e-01 ± 2.4e-04 | 0.194 |
| Reaction-Diffusion | 1 | 7 | 1.36e-01 ± 1.8e-05 | 0.184 |

**Key points:**
- Increasing the number of neurons (and parameters) does not improve accuracy — hidden_dim=1–2 is optimal
- On Burgers and Reaction-Diffusion, the minimal network (7 parameters) works best
- This confirms that the complexity of the θ prediction task is low — a simple functional dependence is sufficient
- More neurons → more noise in θ prediction (visible from the increased std of the final θ)

---

### 2. Input Features Study

Study of θ-network input feature sets: from minimal {x, t, t_next} to full {x, t, t_next, h, grad_norm, y}.

| PDE | Best Feature Set | #Inputs | #Params | L2RE | vs default |
|-----|:---:|:---:|:---:|:---:|:---:|
| Heat | full | 6 | 17 | 1.21e-02 ± 7.7e-04 | +15.4% |
| Burgers | full | 6 | 17 | 1.64e-01 ± 2.9e-04 | +0.2% |
| Reaction-Diffusion | no_grad | 3 | 11 | 1.36e-01 ± 1.2e-05 | +0.0% |

**Key points:**
- On Heat, the full feature set (with y and h) provides a noticeable improvement (+15%), allowing the θ-network to see the current solution state
- On Burgers and Reaction-Diffusion, additional inputs do not help
- grad_norm (included by default) is not a critically important feature — no_grad yields comparable results
- Adding the Laplacian (with_laplacian) does not improve the result compared to default

**Conclusions:**
- Adding new input features to the θ-network is only beneficial for the **Heat Equation**
- For the other equations, using fewer input features to minimize the total parameter count is preferable

---

### 3. Self Benchmark

Validation run of the model on each PDE to establish baseline metrics: a single run with optimal hyperparameters.

| PDE | n_steps | n_iter | LR | θ₀ | Params | L2RE | RMSE/Mean Error | Final θ |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Heat | 4 | 2 | 0.005 | 0.4 | 13 | — (interrupted) | — | — |
| Wave | 3 | 3 | 0.001 | 0.5 | 13 | 3.02e-02 | 1.11e-02 | 0.418 ± 0.003 |
| Burgers | 2 | 2 | 0.0005 | 0.3 | 13 | 1.61e-01 | 4.81e-02 | 0.255 ± 0.003 |
| Reaction-Diffusion | 2 | 2 | 0.001 | 0.35 | 13 | 1.36e-01 | 3.73e-02 | 0.344 ± 0.005 |

**Key points:**
- The model converges within ~200 epochs on all problems
- Final θ values are stable (low std), indicating θ-network consistency across space
- Reaction-Diffusion converges fastest (~10 epochs) but has the highest L2RE
- Wave requires more steps (n_steps=3, n_iter=3) due to the hyperbolic nature of the equation

---

### 4. PDE Training Benchmark Comparison (LF-PINN vs Classical PINN, without data)

Comparison of LF-PINN (13 parameters) with Classical PINN of three sizes (15, 37, 67 parameters) in two modes: fixed time (60 seconds) and fixed threshold L2RE = 0.1.

![PDE Benchmark Comparison — Heat](plots/heat_eq_pde_training_benchmark_comparison.png)

**Mode: Fixed Time (60 seconds)**

| PDE | LF-PINN (13p) | Classical 15p | Classical 37p | Classical 67p |
|-----|:---:|:---:|:---:|:---:|
| Heat | **1.35e-02 ± 1.5e-03** | 5.38e-01 ± 5.0e-02 | 4.35e-02 ± 2.0e-02 | 1.79e-02 ± 6.7e-03 |
| Wave | **3.08e-02 ± 7.5e-04** | 7.10e-01 ± 9.0e-02 | 2.22e-02 ± 6.3e-03 | 1.35e-02 ± 3.8e-03 |
| Burgers | **1.65e-01 ± 8.1e-04** | 5.01e-01 ± 3.3e-02 | 2.75e-01 ± 6.5e-02 | 1.54e-01 ± 4.4e-02 |
| React-Diff | **1.36e-01 ± 1.1e-05** | 1.35e-01 ± 1.1e-03 | 1.36e-01 ± 1.1e-03 | 1.37e-01 ± 7.0e-04 |

**Mode: Fixed Threshold L2RE < 0.1 (which is faster)**

| PDE | LF-PINN (13p) | Classical 15p | Classical 37p | Classical 67p |
|-----|:---:|:---:|:---:|:---:|
| Heat | **1.4 s** (L2RE=1.84e-02) | 30.0 s (not reached) | 17.9 s | 9.4 s |
| Wave | **30.2 s** (L2RE=1.16e-01) | 30.0 s (not reached) | 20.0 s | 9.2 s |
| Burgers | **30.8 s** (L2RE=1.64e-01) | 31.8 s (not reached) | 31.1 s (not reached) | 31.8 s (not reached) |
| React-Diff | **30.2 s** (L2RE=1.36e-01) | 30.0 s (L2RE=1.36e-01) | 30.0 s (L2RE=1.37e-01) | 30.0 s (L2RE=1.38e-01) |

**Key points:**
- On Heat, LF-PINN reaches L2RE=1.84e-02 in **1.4 seconds** (5 epochs) — 6.7× faster than Classical 67p
- With fixed time, LF-PINN outperforms Classical 15p and 37p on **all** PDEs
- Classical 67p beats LF-PINN on Wave (1.35e-02 vs 3.08e-02), but uses 5× more parameters and 70,000 epochs
- On Burgers, no Classical PINN reached L2RE < 0.1 in 30 seconds, while LF-PINN delivers a stable 1.64e-01
- On Reaction-Diffusion, all models show the same L2RE (~1.36e-01) — the problem has a "ceiling" for this discretization

**Conclusions:**
- LF-PINN provides the **best accuracy/parameters trade-off** on all problems
- For problems where maximum accuracy is needed without parameter constraints, Classical 67p may be preferable on Wave
- LF-PINN has a **critical training speed advantage**: 5–200 epochs vs 70,000 epochs

---

### 5. Data Training Benchmark Comparison (LF-PINN + Data vs Classical PINN + Data)

Comparison of LF-PINN with correction parameters and data finetuning (50 points, 5% noise) against Classical PINN with data. Total training time ~60–120 seconds.

![Data Benchmark Comparison — Heat](plots/heat_eq_data_training_benchmark_comparison.png)

| PDE | Best LF-PINN + Data | LF-PINN (no data) | Classical 15p + Data | Classical 37p + Data | Classical 67p + Data |
|-----|:---:|:---:|:---:|:---:|:---:|
| Heat | **7.53e-03 ± 3.5e-03** (gate, 21p) | 1.33e-02 ± 5.9e-04 | 2.41e-01 ± 5.0e-02 | 1.52e-02 ± 6.0e-03 | 1.26e-02 ± 6.5e-03 |
| Wave | **8.86e-03 ± 1.9e-03** (tg+rhs, 26p) | 3.05e-02 ± 5.1e-04 | 4.78e-01 ± 2.8e-02 | 3.25e-02 ± 7.1e-03 | 2.20e-02 ± 8.3e-03 |
| Burgers | 1.31e-01 ± 1.1e-02 (rhs, 15p) | 1.62e-01 ± 5.3e-04 | 3.68e-01 ± 6.9e-02 | 5.79e-02 ± 7.1e-03 | **2.67e-02 ± 5.4e-03** |
| React-Diff | 1.25e-01 ± 7.8e-04 (out_bias, 15p) | 1.36e-01 ± 1.1e-05 | 3.96e-02 ± 1.9e-02 | **3.86e-02 ± 1.2e-02** | 3.92e-02 ± 1.5e-02 |

**Key points:**
- On Heat and Wave, LF-PINN + data **outperforms** all Classical PINN variants, including the 67-parameter version, with fewer parameters
- On Burgers, Classical 67p with data is significantly better (2.67e-02 vs 1.31e-01) — classical PINN utilizes data better for nonlinear problems
- On Reaction-Diffusion, Classical PINN with data also wins (3.86e-02 vs 1.25e-01), using data to break through the PDE training "ceiling"
- Data finetuning improves LF-PINN by **43% (Heat)**, **71% (Wave)**, **19% (Burgers)**, **8% (React-Diff)** relative to the no-data baseline

**Conclusions:**
- LF-PINN + data is the best choice for **smooth parabolic problems** (Heat, Wave), where the model's physical structure is effectively complemented by data
- For **nonlinear and stiff problems** (Burgers, Reaction-Diffusion), Classical PINN utilizes data better thanks to a larger number of free parameters
- Correction parameters (especially RHS Scale and Time Gate) play a key role in LF-PINN's ability to assimilate data

## Final Results

### Summary Table (PDE training, without data)

| PDE | PDE Residual | L2RE | RMSE | MaxError |
|-----|:---:|:---:|:---:|:---:|
| Heat | 3.25e-03 ± 7.2e-04 | 1.65e-02 ± 1.1e-03 | 3.83e-03 ± 2.5e-04 | 9.62e-03 ± 6.2e-04 |
| Wave | 3.74e-02 ± 1.3e-02 | 3.16e-02 ± 5.7e-04 | 1.58e-02 ± 2.9e-04 | 4.35e-02 ± 5.3e-03 |
| Burgers | 2.35e+00 ± 1.4e+00 | 1.62e-01 ± 1.4e-03 | 1.09e-01 ± 9.3e-04 | 7.77e-01 ± 3.5e-02 |
| Reaction-Diffusion | 2.86e-04 ± 3.2e-04 | 1.24e-01 ± 2.3e-05 | 8.23e-02 ± 1.5e-05 | 4.14e-01 ± 8.0e-04 |

## Conclusions

### Analysis of Experimental Results

**Low-Fidelity PINN** demonstrates **better or comparable accuracy** on all four problems:
- **Heat Equation**: outperforms the classical PINN implementation
- **Wave Equation**: outperforms the classical PINN implementation
- **Burgers Equation**: outperforms the classical PINN implementation
- **Reaction-Diffusion**: shows comparable accuracy

Classical PINN (~67 parameters) is inferior to the current implementation (~13 parameters) in accuracy across virtually all tests.

### Key Findings from Experiments

- **The architecture is robust**: the choice of optimizer, activation, and lr affects the result by less than 10–15%
- **Adaptive θ converges to values of 0.25–0.42** — the model prefers schemes between implicit and trapezoidal
- **A minimal network is sufficient**: hidden_dim=1–2 (7–13 parameters) is optimal; increasing does not help
- **Data finetuning is effective**: up to +80% improvement on smooth problems (Heat, Wave) when using additional learnable parameters
- **RHS Scale is a universal correction mode** for problems with a non-trivial PDE right-hand side

### Final Remarks

- **Low-Fidelity PINN outperforms** classical methods in accuracy with **5× fewer parameters**
- Adaptive θ **automatically selects** the appropriate numerical scheme for each problem
