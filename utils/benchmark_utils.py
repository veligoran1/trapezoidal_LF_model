import torch
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from model.lf_model import LowFidelityPINN
from model.classic_pinn_model import ClassicalPINN 
from model.lf_model_test_activation import LFPinn_Activation_Test
from model.lf_model_test_initial_theta import LFPinn_InitialTheta_Test
from model.lf_model_test_extra_parameter import LFPinn_InputFeatures_Test
from model.lf_model_test_loss_weighting import LFPinn_LossWeighting_Test
from model.lf_model_test_correction_param import LFPinn_ThetaParams

from utils.pde_utils import evaluate, get_exact_solution_parametric, train_universal
from utils.plot_utils import plot, plot_theta_field, analyze_and_plot
from utils.data_utils import generate_synthetic_data


from utils.config import PDE_CONFIGS, DOMAINS, TITLES, FEATURE_SETS


# ============================================================
# SINGLE PDE TESTS
# ============================================================

def test_heat(n_steps, n_iterations, lr, epochs):
    """Test for the Heat equation"""
    print("\n" + "="*60)
    print("TEST: Heat Equation")
    print("="*60)
    
    model = LowFidelityPINN('heat', n_steps=n_steps, theta_hidden_dim=2, n_iterations=n_iterations, lr = lr, initial_theta = 0.4)
    domain = DOMAINS['heat']
    exact_sol = get_exact_solution_parametric('heat')
    
    history = train_universal(model, domain, epochs=epochs, n_collocation=30, lr=lr)
    results = evaluate(model, domain, exact_solution=exact_sol)
    
    if isinstance(results['theta_statistics'], dict):
        theta_stats = results['theta_statistics']
        print(f"\nResults:")
        print(f"  θ (mean): {theta_stats['mean']:.4f} ± {theta_stats['std']:.4f}")
        print(f"  θ (range): [{theta_stats['min']:.4f}, {theta_stats['max']:.4f}]")
    
    print(f"  PDE residual: {results['pde_residual']:.2e}")
    if 'rmse' in results:
        print(f"  RMSE error: {results['rmse']:.2e}")
        print(f"  L2RE error: {results['l2re']:.2e}")
    
    plot(results, history, "Heat Equation - Low-Fidelity PINN")
    plot_theta_field(model, domain, "Heat Equation")
    
    return results, history


def test_wave(n_steps, n_iterations, lr, epochs):
    """Test for the Wave equation"""
    print("\n" + "="*60)
    print("TEST: Wave Equation")
    print("="*60)
    
    model = LowFidelityPINN('wave', n_steps=n_steps, theta_hidden_dim=2, n_iterations=n_iterations, lr = lr, initial_theta = 0.5)
    domain = DOMAINS['wave']
    exact_sol = get_exact_solution_parametric('wave')
    
    history = train_universal(model, domain, epochs=epochs, n_collocation=30, lr=lr)
    results = evaluate(model, domain, exact_solution=exact_sol)
    
    if isinstance(results['theta_statistics'], dict):
        theta_stats = results['theta_statistics']
        print(f"\nResults:")
        print(f"  θ (mean): {theta_stats['mean']:.4f} ± {theta_stats['std']:.4f}")
        print(f"  θ (range): [{theta_stats['min']:.4f}, {theta_stats['max']:.4f}]")
    
    print(f"  PDE residual: {results['pde_residual']:.2e}")
    if 'rmse' in results:
        print(f"  Mean error: {results['rmse']:.2e}")
        print(f"  Max error: {results['l2re']:.2e}")
    
    plot(results, history, "Wave Equation - Low-Fidelity PINN")
    plot_theta_field(model, domain, "Wave Equation")
    
    return results, history


def test_burgers(n_steps, n_iterations, lr, epochs):
    """Test for the Burgers equation"""
    print("\n" + "="*60)
    print("TEST: Burgers Equation")
    print("="*60)
    
    model = LowFidelityPINN('burgers', n_steps=n_steps, theta_hidden_dim=2, n_iterations=n_iterations, lr = lr, initial_theta = 0.3)
    domain = DOMAINS['burgers'] 
    exact_sol = get_exact_solution_parametric('burgers')
    
    history = train_universal(model, domain, epochs=epochs, n_collocation=30, lr=lr)
    results = evaluate(model, domain, exact_solution=exact_sol)
    
    if isinstance(results['theta_statistics'], dict):
        theta_stats = results['theta_statistics']
        print(f"\nResults:")
        print(f"  θ (mean): {theta_stats['mean']:.4f} ± {theta_stats['std']:.4f}")
        print(f"  θ (range): [{theta_stats['min']:.4f}, {theta_stats['max']:.4f}]")
    
    print(f"  PDE residual: {results['pde_residual']:.2e}")
    if 'rmse' in results:
        print(f"  Mean error: {results['rmse']:.2e}")
        print(f"  Max error: {results['l2re']:.2e}")
    
    plot(results, history, "Burgers Equation - Low-Fidelity PINN")
    plot_theta_field(model, domain, "Burgers Equation")
    
    return results, history


def test_reaction_diffusion(n_steps, n_iterations, lr, epochs):
    """Test for the Reaction-Diffusion equation"""
    print("\n" + "="*60)
    print("TEST: Reaction-Diffusion Equation")
    print("="*60)
    
    model = LowFidelityPINN('reaction_diffusion', n_steps=n_steps, theta_hidden_dim=2, n_iterations=n_iterations, lr = lr, initial_theta = 0.35)
    domain = DOMAINS['reaction_diffusion']
    exact_sol = get_exact_solution_parametric('reaction_diffusion')
    
    history = train_universal(model, domain, epochs=epochs, n_collocation=30, lr=lr)
    results = evaluate(model, domain, exact_solution=exact_sol)
    
    if isinstance(results['theta_statistics'], dict):
        theta_stats = results['theta_statistics']
        print(f"\nResults:")
        print(f"  θ (mean): {theta_stats['mean']:.4f} ± {theta_stats['std']:.4f}")
        print(f"  θ (range): [{theta_stats['min']:.4f}, {theta_stats['max']:.4f}]")
    
    print(f"  PDE residual: {results['pde_residual']:.2e}")
    if 'rmse' in results:
        print(f"  RMSE error: {results['rmse']:.2e}")
        print(f"  L2RE error: {results['l2re']:.2e}")
    
    plot(results, history, "Reaction-Diffusion - Low-Fidelity PINN")
    plot_theta_field(model, domain, "Reaction-Diffusion")
    
    return results, history


# ============================================================
# RUN ALL TESTS
# ============================================================

def run_all_tests():
    """Run all PDE tests"""
    print("\n" + "="*60)
    print("RUNNING ALL Low-Fidelity PINN TESTS")
    print("="*60)
    
    tests = [
        ('heat', test_heat),
        ('wave', test_wave),
        ('burgers', test_burgers),
        ('reaction_diffusion', test_reaction_diffusion),
    ]
    
    results_dict = {}
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running test: {name.upper()}")
            print(f"{'='*60}")
            results, history = test_func()
            results_dict[name] = {'results': results, 'history': history}
        except Exception as e:
            print(f"\n❌ ERROR in test {name}: {e}")
            import traceback
            traceback.print_exc()
            results_dict[name] = None
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'PDE':<20} {'Mean Error':<15} {'L2 Error':<15} {'Max Error':<15} {'PDE Residual':<15} {'θ mean':<10}")
    print("-"*100)
    
    for name, data in results_dict.items():
        if data is None:
            print(f"{name:<20} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15} {'FAILED':<10}")
        else:
            res = data['results']
            mean_err = res.get('rmse', 'N/A')
            l2_err = res.get('l2_error', 'N/A')
            max_err = res.get('l2re', 'N/A')
            pde_res = res.get('pde_residual', 'N/A')
            theta = res.get('theta_statistics', {})
            theta_mean = theta.get('mean', 'N/A') if isinstance(theta, dict) else 'N/A'
            
            mean_err_str = f"{mean_err:.2e}" if isinstance(mean_err, float) else str(mean_err)
            l2_err_str = f"{l2_err:.2e}" if isinstance(l2_err, float) else str(l2_err)
            max_err_str = f"{max_err:.2e}" if isinstance(max_err, float) else str(max_err)
            pde_res_str = f"{pde_res:.2e}" if isinstance(pde_res, float) else str(pde_res)
            theta_str = f"{theta_mean:.3f}" if isinstance(theta_mean, float) else str(theta_mean)
            
            print(f"{name:<20} {mean_err_str:<15} {l2_err_str:<15} {max_err_str:<15} {pde_res_str:<15} {theta_str:<10}")
    
    print("="*100)
    return results_dict


# ============================================================
# ACTIVATION FUNCTION TESTING
# ============================================================

def test_activations_single_pde(pde_type, n_runs=3, epochs=500, max_time=1000):
    """
    Test different activation functions for one PDE.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        n_runs: number of runs per activation
        epochs: max number of epochs
        max_time: max training time (seconds)
    """
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    
    # List of activations to test
    activations = ['tanh', 'softplus', 'gelu', 'leakyrelu_001', 'leakyrelu_005', 'leakyrelu_01', 'leakyrelu_02', 'elu']
    
    print(f"\n{'='*80}")
    print(f"ACTIVATION TESTING: {title}")
    print(f"Activations: {activations}")
    print(f"Runs per activation: {n_runs}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for activation in activations:
        print(f"\n{'='*60}")
        print(f"Testing activation: {activation.upper()}")
        print(f"{'='*60}")
        
        activation_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model with selected activation
            model = LFPinn_Activation_Test(
                pde_type,
                n_steps=config['n_steps'],
                n_iterations=config['n_iterations'],
                lr=config['lr'],
                initial_theta = config['theta'],
                theta_hidden_dim=2,
                activation=activation
            )
            
            # Train
            history = train_universal(model, domain, epochs=epochs, n_collocation=30, max_time=max_time, lr=config['lr'])
            
            # Evaluate
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['converged'] = history['converged']
            
            if hasattr(model, 'get_theta_statistics'):
                theta_stats = model.get_theta_statistics(domain)
                res['theta_mean'] = theta_stats['mean']
                res['theta_std'] = theta_stats['std']
            
            activation_results.append(res)
            print(f"    ✓ L2RE={res['l2re']:.4e}, RMSE={res['rmse']:.4e}, Time={res['training_time']:.1f}s")
        
        all_results[activation] = activation_results
    
    return all_results, domain, title


# ============================================================
# COMPARISON TESTS (LF-PINN vs Classical PINN)
# ============================================================

def run_single_test_comparison(pde_type, domain, exact_sol, seed, 
                               lf_epochs=100, classical_epochs=1000,
                               max_time=None, target_metric=None, target_value=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    
    # Model factories with Classical PINNs of different sizes
    model_factories = {
        'learnable_theta': lambda: LowFidelityPINN(
            pde_type, 
            n_steps=config['n_steps'],
            n_iterations=config['n_iterations'],
            lr=config['lr'],
            initial_theta=config['theta'],
            theta_hidden_dim=2
        ),
        'Ordinary_PINN_15P': lambda: ClassicalPINN(pde_type, hidden_dim=2),
        'Ordinary_PINN_37P': lambda: ClassicalPINN(pde_type, hidden_dim=4),
        'Ordinary_PINN_67P': lambda: ClassicalPINN(pde_type, hidden_dim=6),
    }
    
    # Epochs for each model type
    epochs_map = {
        'learnable_theta': lf_epochs,
        'Ordinary_PINN_15P': classical_epochs,
        'Ordinary_PINN_37P': classical_epochs,
        'Ordinary_PINN_67P': classical_epochs,
    }
    
    # Learning rates for each model type
    lr_map = {
        'learnable_theta': config['lr'],
        'Ordinary_PINN_15P': 0.001,
        'Ordinary_PINN_37P': 0.001,
        'Ordinary_PINN_67P': 0.001,
    }
    
    results = {}
    
    for key, factory in model_factories.items():
        display_names = {
            'learnable_theta': 'Modified_LF_PINN',
            'Ordinary_PINN_15P': 'Ordinary_PINN_15P',
            'Ordinary_PINN_37P': 'Ordinary_PINN_37P',
            'Ordinary_PINN_67P': 'Ordinary_PINN_67P'
        }
        display_name = display_names.get(key, key)
        
        epochs = epochs_map[key]
        model_lr = lr_map[key]
        print(f"  {display_name} ({epochs} epochs, lr={model_lr})...", end=' ')
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        start_time = time.time()
        model = factory()
        
        n_params = sum(p.numel() for p in model.parameters())
        
        history = train_universal(model, domain, epochs=epochs, n_collocation=30, 
                                max_time=max_time, target_metric=target_metric, 
                                target_value=target_value, lr=model_lr)
        
        res = evaluate(model, domain, exact_solution=exact_sol)
        res['history'] = history
        res['training_time'] = history['training_time']
        res['epochs_completed'] = history['epochs_completed']
        res['converged'] = history['converged']
        res['n_params'] = n_params
            
        if hasattr(model, 'get_theta_statistics'):
            theta_stats = model.get_theta_statistics(domain)
            res['theta_mean'] = theta_stats['mean']
            res['theta_std'] = theta_stats['std']
        
        results[key] = res
        print(f"✓ L2RE={res['l2re']:.4e}, Params={n_params}")
    
    return results


def run_multiple_tests(pde_type, n_runs, lf_epochs=100, classical_epochs=1000,
                       max_time=None, target_value=None, target_metric=None):
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5})
    
    print(f"\n{'='*80}")
    print(f"TESTING: {title} ({n_runs} runs)")
    print(f"Configuration: n_steps={config['n_steps']}")
    print(f"Epochs: LF-PINN={lf_epochs}, Classical={classical_epochs}")
    print(f"{'='*80}\n")
    
    seeds = [42 + i*111 for i in range(n_runs)]
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (seed={seed}):")
        results = run_single_test_comparison(
            pde_type, domain, exact_sol, seed,
            lf_epochs=lf_epochs,
            classical_epochs=classical_epochs,
            max_time=max_time,
            target_value=target_value,
            target_metric=target_metric
        )
        all_results.append(results)
    
    analyze_and_plot(all_results, title, domain)
    return all_results


# ============================================================
# LEARNING RATE STUDY
# ============================================================

def test_lr_single_pde(pde_type: str, 
                       lr_values: List[float] = None,
                       n_runs: int = 5,
                       max_time: float = None,
                       epochs: int = None,
                       n_steps: int = 2,
                       n_iterations: int = 2,
                       n_collocation: int = 30,
                       verbose: bool = True) -> Dict:
    """
    Test different learning rates for one PDE type.
    
    Args:
        pde_type: equation type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        lr_values: list of lr values to test
        n_runs: number of runs per lr
        max_time: max training time (seconds) - if specified
        epochs: number of epochs - if specified (priority over max_time)
        n_steps: number of steps for LF-PINN
        n_collocation: number of collocation points
        verbose: print progress
    
    Returns:
        Dict with results for each lr
    """
    if lr_values is None:
        lr_values = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05]
    
    # Need to specify either max_time or epochs
    if max_time is None and epochs is None:
        epochs = 300  # default
    
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    results = {}
    
    for lr in lr_values:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Testing LR = {lr}")
            print(f"{'='*50}")
        
        lr_results = {
            'l2re': [],
            'rmse': [],
            'final_loss': [],
            'epochs': [],
            'training_time': [],
            'histories': []
        }
        
        for run in range(n_runs):
            if verbose:
                print(f"  Run {run+1}/{n_runs}...", end=" ")
            
            seed = 42 + 111 * run
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = LowFidelityPINN(pde_type, n_steps=config['n_steps'], n_iterations=config['n_iterations'], theta_hidden_dim=2, initial_theta=config['theta'], lr=lr)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            history = {'losses': [], 'pde_losses': []}
            start_time = time.time()
            epoch = 0
            
            # Stopping condition: either by epochs or by time
            while True:
                if epochs is not None and epoch >= epochs:
                    break
                if max_time is not None and time.time() - start_time >= max_time:
                    break
                
                optimizer.zero_grad()
                total_loss, loss_dict = model.total_loss(domain, n_collocation)
                
                if torch.isnan(total_loss):
                    if verbose:
                        print("NaN!", end=" ")
                    break
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                history['losses'].append(loss_dict['total'])
                history['pde_losses'].append(loss_dict['pde'])
                epoch += 1
            
            training_time = time.time() - start_time
            
            res = evaluate(model, domain, exact_solution=exact_sol)
            
            lr_results['l2re'].append(res.get('l2re', float('nan')))
            lr_results['rmse'].append(res.get('rmse', float('nan')))
            lr_results['final_loss'].append(history['losses'][-1] if history['losses'] else float('nan'))
            lr_results['epochs'].append(epoch)
            lr_results['training_time'].append(training_time)
            lr_results['histories'].append(history)
            
            if verbose:
                print(f"L2RE={res.get('l2re', 0):.2e}, epochs={epoch}, time={training_time:.1f}s")
        
        # Statistics
        lr_results['l2re_mean'] = np.nanmean(lr_results['l2re'])
        lr_results['l2re_std'] = np.nanstd(lr_results['l2re'])
        lr_results['rmse_mean'] = np.nanmean(lr_results['rmse'])
        lr_results['rmse_std'] = np.nanstd(lr_results['rmse'])
        lr_results['epochs_mean'] = np.mean(lr_results['epochs'])
        lr_results['time_mean'] = np.mean(lr_results['training_time'])
        lr_results['final_loss_mean'] = np.nanmean(lr_results['final_loss'])
        
        results[lr] = lr_results
        
        if verbose:
            print(f"  → L2RE: {lr_results['l2re_mean']:.2e} ± {lr_results['l2re_std']:.2e}")
    
    return results


# ============================================================
# DATA FINETUNING LEARNING RATE STUDY
# ============================================================

def test_lr_data_finetuning(pde_type: str,
                            correction_mode: str = 'none',
                            lr_finetune_values: List[float] = None,
                            n_runs: int = 5,
                            theta_hidden_dim: int = 2,
                            freeze_base: bool = True,
                            # Pretrain parameters
                            pretrain_epochs: int = None,
                            pretrain_time: float = None,
                            # Finetune parameters
                            finetune_epochs: int = None,
                            finetune_time: float = None,
                            # Data parameters
                            n_data: int = 50,
                            noise_level: float = 0.05,
                            n_collocation: int = 30,
                            # Time anchors (для time-based режимов)
                            n_time_anchors: int = 5,
                            verbose: bool = True) -> Dict:
    """
    Learning rate study for LF-PINN finetuning on data с поддержкой correction_mode.
    
    Process:
        1. Pretrain на PDE (corrections отключены)
        2. Для каждого lr: загрузить веса → finetune на данных (corrections включены)
    
    Args:
        correction_mode: режим коррекции из LFPinn_ThetaParams.AVAILABLE_MODES
                         'none' — baseline, дообучаем theta_net
                         'per_step_bias', 'output_bias', 'rhs_scale',
                         'step_bias+output_bias', 'time_gate+rhs_scale', и т.д.
        freeze_base: заморозить theta_net при finetune (не для 'none')
    """
    
    if lr_finetune_values is None:
        lr_finetune_values = [0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    if pretrain_epochs is None and pretrain_time is None:
        pretrain_epochs = 300
    if finetune_epochs is None and finetune_time is None:
        finetune_epochs = 100
    
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    results = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"LR FINETUNING STUDY: {pde_type.upper()} | mode={correction_mode}")
        print(f"{'='*70}")
    
    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN {run+1}/{n_runs}")
            print(f"{'='*60}")
        
        seed = 42 + 111 * run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain, n_points=n_data,
            noise_level=noise_level, seed=seed
        )
        
        # ============================================================
        # PRETRAIN: PDE (corrections OFF)
        # ============================================================
        if verbose:
            mode_str = f"{pretrain_epochs} epochs" if pretrain_epochs else f"{pretrain_time}s"
            print(f"\n  [Pretrain] lr={config['lr']}, {mode_str}...")
        
        model = LFPinn_ThetaParams(
            pde_type,
            correction_mode=correction_mode,
            n_steps=config['n_steps'],
            theta_hidden_dim=theta_hidden_dim,
            n_iterations=config['n_iterations'],
            initial_theta=config['theta'],
            lr=config['lr'],
            n_time_anchors=n_time_anchors,
            t_max=domain['t'][1]
        )
        
        model.disable_corrections()
        model.freeze_corrections()
        
        device = next(model.parameters()).device
        data_points = data_points.to(device)
        data_values = data_values.to(device)
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr']
        )
        start_time = time.time()
        pretrain_epoch = 0
        
        while True:
            if pretrain_epochs is not None and pretrain_epoch >= pretrain_epochs:
                break
            if pretrain_time is not None and time.time() - start_time >= pretrain_time:
                break
            
            optimizer.zero_grad()
            total_loss, loss_dict = model.total_loss(domain, n_collocation)
            
            if torch.isnan(total_loss):
                if verbose:
                    print(f"  NaN at epoch {pretrain_epoch}")
                break
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pretrain_epoch += 1
        
        pretrain_actual_time = time.time() - start_time
        
        res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
        if verbose:
            print(f"  [Pretrain] Done: {pretrain_epoch} epochs, {pretrain_actual_time:.1f}s, L2RE={res_pretrain['l2re']:.2e}")
        
        pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # ============================================================
        # FINETUNE: Data (corrections ON, для каждого lr)
        # ============================================================
        for lr in lr_finetune_values:
            if lr not in results:
                results[lr] = {
                    'l2re': [], 'rmse': [], 'l2re_before': [],
                    'improvement': [], 'final_data_loss': [],
                    'finetune_epochs': [], 'finetune_time': [],
                    'histories': []
                }
            
            if verbose:
                print(f"  [Finetune] lr={lr}...", end=" ")
            
            model.load_state_dict({k: v.clone() for k, v in pretrained_state.items()})
            
            # Включаем коррекции
            model.enable_corrections()
            model.unfreeze_corrections()
            
            # Выбираем что обучаем
            if correction_mode == 'none':
                finetune_params = list(model.parameters())
            elif freeze_base:
                model.freeze_base_theta()
                finetune_params = model.get_correction_params()
            else:
                finetune_params = list(model.parameters())
            
            trainable = [p for p in finetune_params if p.requires_grad]
            
            optimizer = torch.optim.Adam(trainable, lr=lr)
            start_time = time.time()
            finetune_epoch = 0
            history = {'data_loss': []}
            
            while True:
                if finetune_epochs is not None and finetune_epoch >= finetune_epochs:
                    break
                if finetune_time is not None and time.time() - start_time >= finetune_time:
                    break
                
                optimizer.zero_grad()
                
                x_data = data_points[:, 0:1]
                t_data = data_points[:, 1:2]
                loss = model.data_loss(x_data, t_data, data_values)
                
                if torch.isnan(loss):
                    if verbose:
                        print("NaN!", end=" ")
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                history['data_loss'].append(loss.item())
                finetune_epoch += 1
            
            finetune_actual_time = time.time() - start_time
            
            res_finetune = evaluate(model, domain, exact_solution=exact_sol)
            
            l2re_before = res_pretrain['l2re']
            l2re_after = res_finetune['l2re']
            improvement = (l2re_before - l2re_after) / l2re_before * 100 if l2re_before > 1e-12 else 0.0
            
            results[lr]['l2re'].append(l2re_after)
            results[lr]['rmse'].append(res_finetune['rmse'])
            results[lr]['l2re_before'].append(l2re_before)
            results[lr]['improvement'].append(improvement)
            results[lr]['final_data_loss'].append(history['data_loss'][-1] if history['data_loss'] else float('nan'))
            results[lr]['finetune_epochs'].append(finetune_epoch)
            results[lr]['finetune_time'].append(finetune_actual_time)
            results[lr]['histories'].append(history)
            
            if verbose:
                print(f"L2RE: {l2re_before:.2e} → {l2re_after:.2e} ({improvement:+.1f}%), {finetune_epoch} epochs")
    
    # Statistics
    for lr in lr_finetune_values:
        r = results[lr]
        r['l2re_mean'] = np.nanmean(r['l2re'])
        r['l2re_std'] = np.nanstd(r['l2re'])
        r['rmse_mean'] = np.nanmean(r['rmse'])
        r['rmse_std'] = np.nanstd(r['rmse'])
        r['improvement_mean'] = np.nanmean(r['improvement'])
        r['improvement_std'] = np.nanstd(r['improvement'])
        r['finetune_epochs_mean'] = np.mean(r['finetune_epochs'])
        r['finetune_time_mean'] = np.mean(r['finetune_time'])
        r['final_data_loss_mean'] = np.nanmean(r['final_data_loss'])
    
    results['baseline'] = {
        'l2re_mean': np.mean([results[lr_finetune_values[0]]['l2re_before'][i] for i in range(n_runs)]),
        'l2re_std': np.std([results[lr_finetune_values[0]]['l2re_before'][i] for i in range(n_runs)]),
        'improvement_mean': 0.0,
        'improvement_std': 0.0
    }
    
    if verbose:
        print(f"\n{'='*90}")
        print(f"SUMMARY: {pde_type.upper()} - LR Study (mode={correction_mode})")
        print(f"{'='*90}")
        print(f"{'LR':<12} {'L2RE After':<24} {'Improvement':<20} {'Epochs':<10} {'Time':<10}")
        print(f"{'-'*90}")
        print(f"{'baseline':<12} {results['baseline']['l2re_mean']:.2e} ± {results['baseline']['l2re_std']:.2e}         {'--':<20} {'--':<10} {'--':<10}")
        
        best_lr = min(lr_finetune_values, key=lambda x: results[x]['l2re_mean'])
        for lr in lr_finetune_values:
            r = results[lr]
            l2re_str = f"{r['l2re_mean']:.2e} ± {r['l2re_std']:.2e}"
            imp_str = f"{r['improvement_mean']:+.1f}% ± {r['improvement_std']:.1f}%"
            marker = " ← BEST" if lr == best_lr else ""
            print(f"{lr:<12.5f} {l2re_str:<24} {imp_str:<20} {r['finetune_epochs_mean']:<10.0f} {r['finetune_time_mean']:<10.1f}s{marker}")
        print(f"{'='*90}")
    
    return results


# ============================================================
# DATA TRAINING EPOCHS SATURATION STUDY
# ============================================================

def data_training_epochs_study(pde_type: str,
                               correction_mode: str = 'none',
                               n_runs: int = 5,
                               theta_hidden_dim: int = 2,
                               freeze_base: bool = True,
                               pretrain_epochs: int = 300,
                               finetune_lr: float = 0.0001,
                               finetune_epochs: int = 500,
                               eval_every: int = 10,
                               n_data: int = 50,
                               noise_level: float = 0.05,
                               n_collocation: int = 30,
                               n_time_anchors: int = 5,
                               verbose: bool = True):
    """
    Study: how many finetune epochs are needed? Tracks L2RE every eval_every epochs.
    
    Args:
        correction_mode: режим коррекции ('none', 'per_step_bias', 'rhs_scale', ...)
        freeze_base: заморозить theta_net при finetune (не для 'none')
    """
    
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    
    all_epochs = []
    all_l2re = []
    all_data_loss = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EPOCHS SATURATION STUDY: {pde_type.upper()} | mode={correction_mode}")
        print(f"{'='*70}")
    
    for run in range(n_runs):
        if verbose:
            print(f"\nRun {run+1}/{n_runs}")
        
        seed = 42 + 111 * run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain, n_points=n_data,
            noise_level=noise_level, seed=seed
        )
        
        # ============================================================
        # PRETRAIN: PDE (corrections OFF)
        # ============================================================
        model = LFPinn_ThetaParams(
            pde_type,
            correction_mode=correction_mode,
            n_steps=config['n_steps'],
            theta_hidden_dim=theta_hidden_dim,
            n_iterations=config['n_iterations'],
            initial_theta=config['theta'],
            lr=config['lr'],
            n_time_anchors=n_time_anchors,
            t_max=domain['t'][1]
        )
        
        model.disable_corrections()
        model.freeze_corrections()
        
        device = next(model.parameters()).device
        data_points = data_points.to(device)
        data_values = data_values.to(device)
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr']
        )
        
        for epoch in range(pretrain_epochs):
            optimizer.zero_grad()
            loss, _ = model.total_loss(domain, n_collocation)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
        if verbose:
            print(f"  Pretrain done: L2RE={res_pretrain['l2re']:.2e}")
        
        # ============================================================
        # FINETUNE: Data (corrections ON, tracking L2RE)
        # ============================================================
        model.enable_corrections()
        model.unfreeze_corrections()
        
        if correction_mode == 'none':
            finetune_params = list(model.parameters())
        elif freeze_base:
            model.freeze_base_theta()
            finetune_params = model.get_correction_params()
        else:
            finetune_params = list(model.parameters())
        
        trainable = [p for p in finetune_params if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=finetune_lr)
        
        x_data = data_points[:, 0:1]
        t_data = data_points[:, 1:2]
        
        run_epochs = [0]
        run_l2re = [res_pretrain['l2re']]
        run_data_loss = [float('nan')]
        
        for epoch in range(1, finetune_epochs + 1):
            optimizer.zero_grad()
            data_loss = model.data_loss(x_data, t_data, data_values)
            
            if torch.isnan(data_loss):
                break
            
            data_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch % eval_every == 0:
                res = evaluate(model, domain, exact_solution=exact_sol)
                run_epochs.append(epoch)
                run_l2re.append(res['l2re'])
                run_data_loss.append(data_loss.item())
                
                if verbose and epoch % (eval_every * 5) == 0:
                    print(f"  Epoch {epoch}: L2RE={res['l2re']:.2e}, DataLoss={data_loss.item():.2e}")
        
        all_epochs.append(run_epochs)
        all_l2re.append(run_l2re)
        all_data_loss.append(run_data_loss)
    
    # Averaging
    max_len = max(len(e) for e in all_epochs)
    epochs_common = all_epochs[0][:max_len]
    
    l2re_matrix = np.array([r + [np.nan]*(max_len - len(r)) for r in all_l2re])
    l2re_mean = np.nanmean(l2re_matrix, axis=0)
    l2re_std = np.nanstd(l2re_matrix, axis=0)
    
    data_loss_matrix = np.array([r + [np.nan]*(max_len - len(r)) for r in all_data_loss])
    data_loss_mean = np.nanmean(data_loss_matrix, axis=0)
    
    if verbose:
        print(f"\n  Final L2RE: {l2re_mean[-1]:.2e} ± {l2re_std[-1]:.2e}")
        best_idx = np.nanargmin(l2re_mean)
        print(f"  Best L2RE: {l2re_mean[best_idx]:.2e} at epoch {epochs_common[best_idx]}")
    
    return {
        'epochs': epochs_common,
        'l2re_mean': l2re_mean,
        'l2re_std': l2re_std,
        'data_loss_mean': data_loss_mean,
        'all_l2re': all_l2re,
        'all_epochs': all_epochs,
        'correction_mode': correction_mode,
        'pde_type': pde_type,
    }


# ============================================================
# INITIAL THETA TESTING
# ============================================================

def test_initial_theta_single_pde(pde_type, 
                                   custom_theta=None, 
                                   n_runs=5, 
                                   epochs=500, 
                                   max_time=None,
                                   # Model parameters (if None - taken from PDE_CONFIGS)
                                   n_steps=None,
                                   n_iterations=None,
                                   lr=None,
                                   n_collocation=30):
    """
    Test different initial theta values for one PDE.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        custom_theta: custom theta value (from previous experiments)
        n_runs: number of runs per theta value
        epochs: max number of epochs
        max_time: max training time (seconds)
        n_steps: number of steps (if None - from PDE_CONFIGS)
        n_iterations: number of iterations (if None - from PDE_CONFIGS)
        lr: learning rate (if None - from PDE_CONFIGS)
        n_collocation: number of collocation points
    """
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    # Get defaults from config, but allow override
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001})
    
    n_steps = n_steps if n_steps is not None else config.get('n_steps', 5)
    n_iterations = n_iterations if n_iterations is not None else config.get('n_iterations', 2)
    lr = lr if lr is not None else config.get('lr', 0.001)
    
    # List of theta values to test
    theta_values = [0.0, 0.5, 1.0]
    theta_names = {0.0: 'Implicit (θ=0)', 0.5: 'Trapezoidal (θ=0.5)', 1.0: 'Explicit (θ=1)'}
    
    if custom_theta is not None:
        theta_values.append(custom_theta)
        theta_names[custom_theta] = f'Custom (θ={custom_theta:.2f})'
    
    print(f"\n{'='*80}")
    print(f"INITIAL THETA VALUE TESTING: {title}")
    print(f"Theta values: {theta_values}")
    print(f"Runs per value: {n_runs}")
    print(f"Parameters: n_steps={n_steps}, n_iterations={n_iterations}, lr={lr}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for theta in theta_values:
        print(f"\n{'='*60}")
        print(f"Testing: {theta_names[theta]}")
        print(f"{'='*60}")
        
        theta_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model with passed parameters
            model = LFPinn_InitialTheta_Test(
                pde_type,
                n_steps=n_steps,
                n_iterations=n_iterations,
                lr=lr,
                theta_hidden_dim=2,
                initial_theta=theta
            )
            
            # Train
            history = train_universal(model, domain, epochs=epochs, 
                                      n_collocation=n_collocation, max_time=max_time, lr=lr)
            
            # Evaluate
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['initial_theta'] = theta
            
            # Final theta value
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
            
            theta_results.append(res)
            print(f"    ✓ L2RE={res['l2re']:.4e}, θ_final={theta_stats['mean']:.3f}±{theta_stats['std']:.3f}")
        
        all_results[theta] = theta_results
    
    return all_results, domain, title


# ============================================================
# OPTIMIZER TESTING
# ============================================================

from model.lf_model_test_optimizer import LFPinn_Optimizer_Test

def test_optimizers_single_pde(pde_type, n_runs=5, epochs=500, max_time=None,
                                n_steps=None, n_iterations=None, lr=None,
                                n_collocation=30):
    """
    Test different optimizers for one PDE.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        n_runs: number of runs per optimizer
        epochs: max number of epochs
        max_time: max training time (seconds)
        n_steps, n_iterations, lr: model parameters (None = from PDE_CONFIGS)
        n_collocation: number of collocation points
    """
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001})
    n_steps = n_steps if n_steps is not None else config.get('n_steps', 5)
    n_iterations = n_iterations if n_iterations is not None else config.get('n_iterations', 2)
    lr = lr if lr is not None else config.get('lr', 0.001)
    
    # Optimizers to test
    optimizers = [
        'adam', 'adamw', 'adamw_wd01', 'adamw_wd001',
        'rmsprop', 'rmsprop_centered'
    ]
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZER TESTING: {title}")
    print(f"Optimizers: {optimizers}")
    print(f"Runs per optimizer: {n_runs}")
    print(f"Parameters: n_steps={n_steps}, n_iterations={n_iterations}, lr={lr}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"Testing optimizer: {opt_name.upper()}")
        print(f"{'='*60}")
        
        opt_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            model = LFPinn_Optimizer_Test(
                pde_type,
                n_steps=n_steps,
                n_iterations=n_iterations,
                lr=lr,
                theta_hidden_dim=2,
                optimizer=opt_name
            )
            
            # Custom training loop using model's optimizer
            optimizer = model.get_optimizer()
            history = {'losses': [], 'pde_losses': []}
            start_time = time.time()
            
            for epoch in range(epochs):
                if max_time is not None and time.time() - start_time >= max_time:
                    break
                
                optimizer.zero_grad()
                total_loss, loss_dict = model.total_loss(domain, n_collocation)
                
                if torch.isnan(total_loss):
                    print(f"    NaN at epoch {epoch}")
                    break
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                history['losses'].append(loss_dict['total'])
                history['pde_losses'].append(loss_dict['pde'])
            
            training_time = time.time() - start_time
            
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = training_time
            res['epochs_completed'] = len(history['losses'])
            res['optimizer'] = opt_name
            
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
            
            opt_results.append(res)
            print(f"    ✓ L2RE={res['l2re']:.4e}, Time={training_time:.1f}s")
        
        all_results[opt_name] = opt_results
    
    return all_results, domain, title

def create_optimizer(name: str, params, lr: float):
    """Create optimizer by name"""
    optimizers = {
        'adam': lambda p, lr: torch.optim.Adam(p, lr=lr),
        'adamw': lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=0.01),
        'adamw_wd001': lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=0.001),
        'rmsprop': lambda p, lr: torch.optim.RMSprop(p, lr=lr, alpha=0.99),
        'rmsprop_centered': lambda p, lr: torch.optim.RMSprop(p, lr=lr, alpha=0.99, centered=True),
        'sgd': lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'sgd_nesterov': lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True),
        'nadam': lambda p, lr: torch.optim.NAdam(p, lr=lr),
        'radam': lambda p, lr: torch.optim.RAdam(p, lr=lr),
        'adamax': lambda p, lr: torch.optim.Adamax(p, lr=lr)
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name](params, lr)


# ============================================================
# TEST OPTIMIZER: DATA
# ============================================================

def test_optimizer_data_finetuning(pde_type: str,
                                    optimizer_names: List[str] = None,
                                    finetune_lr: float = 0.0001,
                                    n_runs: int = 5,
                                    # Model parameters
                                    theta_hidden_dim: int = 2,
                                    # Pretrain parameters (PDE phase)
                                    pretrain_epochs: int = 300,
                                    # Finetune parameters (Data phase)
                                    finetune_epochs: int = 200,
                                    # Data parameters
                                    n_data: int = 50,
                                    noise_level: float = 0.05,
                                    n_collocation: int = 30,
                                    verbose: bool = True) -> Dict:
    """
    Test different optimizers for data finetuning stage.
    
    Process:
    1. For each run: train model on PDE (pretrain) with RMSprop
    2. Save weights
    3. For each optimizer: load weights and finetune on data with fixed lr
    
    Args:
        pde_type: equation type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        optimizer_names: list of optimizer names to test
        finetune_lr: fixed learning rate for finetuning (default 0.0001)
        n_runs: number of runs
        
        # Model
        n_steps: number of LF-PINN steps
        n_iterations: number of fixed-point iterations
        theta_hidden_dim: theta_net hidden layer size
        
        # Pretrain
        pretrain_lr: learning rate for PDE phase
        pretrain_epochs: number of epochs for pretrain
        
        # Finetune
        finetune_epochs: number of epochs for finetune
        
        # Data
        n_data: number of data points
        noise_level: noise level in data
        n_collocation: number of collocation points
        
        verbose: print progress
    
    Returns:
        Dict with results for each optimizer
    """
    if optimizer_names is None:
        optimizer_names = ['adam', 'adamw', 'rmsprop', 'sgd', 'nadam', 'radam']
    
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    results = {}
    
    for run in range(n_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN {run+1}/{n_runs}")
            print(f"{'='*60}")
        
        seed = 42 + 111 * run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate data for this run
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain, n_points=n_data,
            noise_level=noise_level, seed=seed
        )
        
        # ============================================================
        # PRETRAIN: Training on PDE (once per run)
        # ============================================================
        if verbose:
            print(f"\n  [Pretrain] AdaM lr={config['lr']}, {pretrain_epochs} epochs...")
        
        model = LowFidelityPINN(
            pde_type, 
            n_steps=config['n_steps'], 
            n_iterations=config['n_iterations'],
            theta_hidden_dim=theta_hidden_dim,
            lr=config['lr'],
            initial_theta=config['theta']
        )
        
        device = next(model.parameters()).device
        data_points = data_points.to(device)
        data_values = data_values.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        for epoch in range(pretrain_epochs):
            optimizer.zero_grad()
            total_loss, loss_dict = model.total_loss(domain, n_collocation)
            
            if torch.isnan(total_loss):
                if verbose:
                    print(f"  NaN at epoch {epoch}")
                break
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Evaluate after pretrain
        res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
        if verbose:
            print(f"  [Pretrain] Done: L2RE={res_pretrain['l2re']:.2e}")
        
        # Save model state
        pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        # ============================================================
        # FINETUNE: Test each optimizer
        # ============================================================
        for opt_name in optimizer_names:
            if opt_name not in results:
                results[opt_name] = {
                    'l2re': [],
                    'rmse': [],
                    'l2re_before': [],
                    'improvement': [],
                    'final_data_loss': [],
                    'finetune_epochs': [],
                    'finetune_time': [],
                    'histories': []
                }
            
            if verbose:
                print(f"  [Finetune] {opt_name}, lr={finetune_lr}...", end=" ")
            
            # Load pretrained state
            model.load_state_dict({k: v.clone() for k, v in pretrained_state.items()})
            
            # Create optimizer
            optimizer = create_optimizer(opt_name, model.parameters(), finetune_lr)
            
            start_time = time.time()
            history = {'data_loss': [], 'total_loss': []}
            
            for epoch in range(finetune_epochs):
                optimizer.zero_grad()
                
                # Data loss only (or combined - можно настроить)
                x_data = data_points[:, 0:1]
                t_data = data_points[:, 1:2]
                u_pred = model.forward(x_data, t_data)
                data_loss = torch.mean((u_pred - data_values)**2)
                
                if torch.isnan(data_loss):
                    if verbose:
                        print("NaN!", end=" ")
                    break
                
                data_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                history['data_loss'].append(data_loss.item())
            
            finetune_time = time.time() - start_time
            
            # Evaluate after finetuning
            res_finetune = evaluate(model, domain, exact_solution=exact_sol)
            
            l2re_before = res_pretrain['l2re']
            l2re_after = res_finetune['l2re']
            improvement = (l2re_before - l2re_after) / l2re_before * 100
            
            results[opt_name]['l2re'].append(l2re_after)
            results[opt_name]['rmse'].append(res_finetune['rmse'])
            results[opt_name]['l2re_before'].append(l2re_before)
            results[opt_name]['improvement'].append(improvement)
            results[opt_name]['final_data_loss'].append(history['data_loss'][-1] if history['data_loss'] else float('nan'))
            results[opt_name]['finetune_epochs'].append(len(history['data_loss']))
            results[opt_name]['finetune_time'].append(finetune_time)
            results[opt_name]['histories'].append(history)
            
            if verbose:
                print(f"L2RE: {l2re_before:.2e} → {l2re_after:.2e} ({improvement:+.1f}%)")
    
    # Compute statistics
    for opt_name in optimizer_names:
        r = results[opt_name]
        r['l2re_mean'] = np.nanmean(r['l2re'])
        r['l2re_std'] = np.nanstd(r['l2re'])
        r['rmse_mean'] = np.nanmean(r['rmse'])
        r['rmse_std'] = np.nanstd(r['rmse'])
        r['improvement_mean'] = np.nanmean(r['improvement'])
        r['improvement_std'] = np.nanstd(r['improvement'])
        r['finetune_time_mean'] = np.mean(r['finetune_time'])
        r['final_data_loss_mean'] = np.nanmean(r['final_data_loss'])
    
    # Baseline (without finetuning)
    results['baseline'] = {
        'l2re_mean': np.mean([results[optimizer_names[0]]['l2re_before'][i] for i in range(n_runs)]),
        'l2re_std': np.std([results[optimizer_names[0]]['l2re_before'][i] for i in range(n_runs)]),
        'improvement_mean': 0.0,
        'improvement_std': 0.0
    }
    
    # Print summary
    if verbose:
        print_optimizer_summary(results, optimizer_names, pde_type, finetune_lr)
    
    return results


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_optimizer_summary(results: Dict, optimizer_names: List[str], 
                            pde_type: str, lr: float):
    """Print summary table"""
    print(f"\n{'='*95}")
    print(f"SUMMARY: {pde_type.upper()} - Optimizer Data Finetuning (lr={lr})")
    print(f"{'='*95}")
    print(f"{'Optimizer':<18} {'L2RE After':<24} {'Improvement':<20} {'Time (s)':<12}")
    print(f"{'-'*95}")
    
    # Baseline
    bl = results['baseline']
    print(f"{'baseline':<18} {bl['l2re_mean']:.2e} ± {bl['l2re_std']:.2e}         {'--':<20} {'--':<12}")
    
    # Sort by L2RE
    sorted_opts = sorted(optimizer_names, key=lambda x: results[x]['l2re_mean'])
    best_opt = sorted_opts[0]
    
    for opt in sorted_opts:
        r = results[opt]
        l2re_str = f"{r['l2re_mean']:.2e} ± {r['l2re_std']:.2e}"
        imp_str = f"{r['improvement_mean']:+.1f}% ± {r['improvement_std']:.1f}%"
        time_str = f"{r['finetune_time_mean']:.2f}"
        marker = " ★ BEST" if opt == best_opt else ""
        print(f"{opt:<18} {l2re_str:<24} {imp_str:<20} {time_str:<12}{marker}")
    
    print(f"{'='*95}")
    print(f"\n✓ RECOMMENDED OPTIMIZER: {best_opt}")
    print(f"  L2RE: {results[best_opt]['l2re_mean']:.2e} ± {results[best_opt]['l2re_std']:.2e}")
    print(f"  Improvement: {results[best_opt]['improvement_mean']:+.1f}%")


# ============================================================
# THETA HIDDEN DIM TESTING
# ============================================================

def test_theta_hidden_dim_single_pde(pde_type: str,
                                      hidden_dims: List[int] = None,
                                      n_runs: int = 5,
                                      epochs: int = 500,
                                      max_time: float = None,
                                      # Model parameters (None = from PDE_CONFIGS)
                                      n_steps: int = None,
                                      n_iterations: int = None,
                                      lr: float = None,
                                      n_collocation: int = 30,
                                      verbose: bool = True) -> Tuple[Dict, dict, str]:
    """
    Test different theta_hidden_dim values for one PDE.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        hidden_dims: list of theta_hidden_dim values to test
        n_runs: number of runs per hidden_dim
        epochs: max number of epochs
        max_time: max training time (seconds)
        n_steps: number of steps (None = from PDE_CONFIGS)
        n_iterations: number of iterations (None = from PDE_CONFIGS)
        lr: learning rate (None = from PDE_CONFIGS)
        n_collocation: number of collocation points
        verbose: print progress
    
    Returns:
        all_results: dict {hidden_dim: [run_results]}
        domain: domain dict
        title: PDE title
    """
    if hidden_dims is None:
        hidden_dims = [1, 2, 4, 8, 16, 32]
    
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    # Get defaults from config
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001})
    n_steps = n_steps if n_steps is not None else config.get('n_steps', 5)
    n_iterations = n_iterations if n_iterations is not None else config.get('n_iterations', 2)
    lr = lr if lr is not None else config.get('lr', 0.001)
    
    # Calculate parameter counts
    def count_theta_params(h):
        # Linear(4, h): 4*h + h = 5*h
        # Linear(h, 1): h*1 + 1 = h + 1
        return 6 * h + 1
    
    print(f"\n{'='*80}")
    print(f"THETA HIDDEN DIM TESTING: {title}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Parameter counts: {[count_theta_params(h) for h in hidden_dims]}")
    print(f"Runs per value: {n_runs}")
    print(f"Model params: n_steps={n_steps}, n_iterations={n_iterations}, lr={lr}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for hidden_dim in hidden_dims:
        n_params = count_theta_params(hidden_dim)
        print(f"\n{'='*60}")
        print(f"Testing: theta_hidden_dim={hidden_dim} ({n_params} params)")
        print(f"{'='*60}")
        
        dim_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            if verbose:
                print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model with specified theta_hidden_dim
            model = LowFidelityPINN(
                pde_type,
                n_steps=config['n_steps'],
                n_iterations=config['n_iterations'],
                lr=config['lr'],
                initial_theta = config['theta'],
                theta_hidden_dim=hidden_dim
            )
            
            # Train
            history = train_universal(
                model, domain, 
                epochs=epochs, 
                n_collocation=n_collocation, 
                max_time=max_time,
                lr = config['lr']
            )
            
            # Evaluate
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['theta_hidden_dim'] = hidden_dim
            res['n_params'] = n_params
            
            # Theta statistics
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
            
            dim_results.append(res)
            
            if verbose:
                print(f"    ✓ L2RE={res['l2re']:.4e}, RMSE={res['rmse']:.4e}, "
                      f"θ={theta_stats['mean']:.3f}±{theta_stats['std']:.3f}, "
                      f"Time={res['training_time']:.1f}s")
        
        all_results[hidden_dim] = dim_results
    
    # Print summary
    if verbose:
        print_theta_hidden_dim_summary(all_results, hidden_dims, pde_type)
    
    return all_results, domain, title


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_theta_hidden_dim_summary(results: Dict, hidden_dims: List[int], pde_type: str):
    """Print summary table"""
    
    def count_theta_params(h):
        return 6 * h + 1
    
    print(f"\n{'='*100}")
    print(f"SUMMARY: {pde_type.upper()} - Theta Hidden Dim Study")
    print(f"{'='*100}")
    print(f"{'Hidden Dim':<12} {'Params':<10} {'L2RE':<24} {'RMSE':<24} {'θ final':<16} {'Time (s)':<10}")
    print(f"{'-'*100}")
    
    # Collect stats
    stats = []
    for h in hidden_dims:
        runs = results[h]
        l2res = [r['l2re'] for r in runs]
        rmses = [r['rmse'] for r in runs]
        thetas = [r['theta_final_mean'] for r in runs]
        times = [r['training_time'] for r in runs]
        
        stats.append({
            'hidden_dim': h,
            'n_params': count_theta_params(h),
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'theta_mean': np.mean(thetas),
            'theta_std': np.std(thetas),
            'time_mean': np.mean(times)
        })
    
    # Find best
    best_idx = np.argmin([s['l2re_mean'] for s in stats])
    
    for i, s in enumerate(stats):
        l2re_str = f"{s['l2re_mean']:.2e} ± {s['l2re_std']:.2e}"
        rmse_str = f"{s['rmse_mean']:.2e} ± {s['rmse_std']:.2e}"
        theta_str = f"{s['theta_mean']:.3f} ± {s['theta_std']:.3f}"
        marker = " ★ BEST" if i == best_idx else ""
        
        print(f"{s['hidden_dim']:<12} {s['n_params']:<10} {l2re_str:<24} {rmse_str:<24} {theta_str:<16} {s['time_mean']:<10.1f}{marker}")
    
    print(f"{'='*100}")
    
    best = stats[best_idx]
    print(f"\n✓ OPTIMAL: theta_hidden_dim={best['hidden_dim']} ({best['n_params']} params)")
    print(f"  L2RE: {best['l2re_mean']:.2e} ± {best['l2re_std']:.2e}")


# ============================================================
# TEST EXTRA FEATURE
# ============================================================

def test_input_features_single_pde(pde_type: str,
                                    feature_sets: Dict[str, List[str]] = None,
                                    n_runs: int = 5,
                                    epochs: int = 500,
                                    max_time: float = None,
                                    # Model parameters
                                    n_steps: int = None,
                                    n_iterations: int = None,
                                    lr: float = None,
                                    theta_hidden_dim: int = 2,
                                    n_collocation: int = 30,
                                    verbose: bool = True) -> Tuple[Dict, dict, str]:
    """
    Test different input feature combinations for theta_net.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        feature_sets: dict {name: [features]} to test (None = use FEATURE_SETS)
        n_runs: number of runs per feature set
        epochs: max number of epochs
        max_time: max training time (seconds)
        n_steps, n_iterations, lr: model parameters (None = from PDE_CONFIGS)
        theta_hidden_dim: hidden dimension for theta_net
        n_collocation: number of collocation points
        verbose: print progress
    
    Returns:
        all_results: dict {set_name: [run_results]}
        domain: domain dict
        title: PDE title
    """
    if feature_sets is None:
        # Default: test subset of interesting combinations
        feature_sets = {
            'no_grad': FEATURE_SETS['no_grad'],
            'default': FEATURE_SETS['default'],
            'with_y': FEATURE_SETS['with_y'],
            'with_t_norm': FEATURE_SETS['with_t_norm'],
            'with_y_abs':FEATURE_SETS['with_y'],
            'with_laplacian':FEATURE_SETS['with_laplacian'],
            'full': FEATURE_SETS['full']
        }
    
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    # Get defaults from config
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001})
    n_steps = n_steps if n_steps is not None else config.get('n_steps', 5)
    n_iterations = n_iterations if n_iterations is not None else config.get('n_iterations', 2)
    lr = lr if lr is not None else config.get('lr', 0.001)
    
    # Calculate parameter counts
    def count_params(n_inputs, hidden_dim):
        # Linear(n_inputs, hidden): n_inputs * hidden + hidden
        # Linear(hidden, 1): hidden + 1
        return (n_inputs + 1) * hidden_dim + hidden_dim + 1
    
    print(f"\n{'='*80}")
    print(f"INPUT FEATURES TESTING: {title}")
    print(f"Feature sets: {list(feature_sets.keys())}")
    print(f"Runs per set: {n_runs}")
    print(f"Model params: n_steps={n_steps}, n_iterations={n_iterations}, lr={lr}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for set_name, features in feature_sets.items():
        n_inputs = len(features)
        n_params = count_params(n_inputs, theta_hidden_dim)
        
        print(f"\n{'='*60}")
        print(f"Testing: {set_name}")
        print(f"Features ({n_inputs}): {features}")
        print(f"Theta params: {n_params}")
        print(f"{'='*60}")
        
        set_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            if verbose:
                print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model with specified features
            t_max = domain['t'][1]
            model = LFPinn_InputFeatures_Test(
                pde_type,
                input_features=features,
                n_steps=config['n_steps'],
                n_iterations=config['n_iterations'],
                lr=config['lr'],
                initial_theta=config['theta'],
                theta_hidden_dim=theta_hidden_dim,
                t_max=t_max
            )
            
            # Train
            start_time = time.time()
            history = train_universal(
                model, domain,
                epochs=epochs,
                n_collocation=n_collocation,
                max_time=max_time,
                lr = config['lr']
            )
            training_time = time.time() - start_time
            
            # Evaluate
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history.get('training_time', training_time)
            res['epochs_completed'] = history['epochs_completed']
            res['feature_set'] = set_name
            res['features'] = features
            res['n_inputs'] = n_inputs
            res['n_params'] = n_params
            
            # Theta statistics
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
            
            set_results.append(res)
            
            if verbose:
                print(f"    ✓ L2RE={res['l2re']:.4e}, RMSE={res['rmse']:.4e}, "
                      f"θ={theta_stats['mean']:.3f}±{theta_stats['std']:.3f}, "
                      f"Time={res['training_time']:.1f}s")
        
        all_results[set_name] = set_results
    
    # Print summary
    if verbose:
        print_input_features_summary(all_results, feature_sets, pde_type)
    
    return all_results, domain, title


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_input_features_summary(results: Dict, feature_sets: Dict, pde_type: str):
    """Print summary table"""
    
    print(f"\n{'='*110}")
    print(f"SUMMARY: {pde_type.upper()} - Input Features Study")
    print(f"{'='*110}")
    print(f"{'Set Name':<15} {'Features':<35} {'#In':<5} {'#P':<6} {'L2RE':<22} {'θ final':<14} {'Time':<8}")
    print(f"{'-'*110}")
    
    # Collect stats
    stats = []
    for name, features in feature_sets.items():
        runs = results[name]
        l2res = [r['l2re'] for r in runs]
        thetas = [r['theta_final_mean'] for r in runs]
        times = [r['training_time'] for r in runs]
        
        stats.append({
            'name': name,
            'features': features,
            'n_inputs': runs[0]['n_inputs'],
            'n_params': runs[0]['n_params'],
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'theta_mean': np.mean(thetas),
            'time_mean': np.mean(times)
        })
    
    # Sort by L2RE
    stats_sorted = sorted(stats, key=lambda x: x['l2re_mean'])
    best_name = stats_sorted[0]['name']
    
    for s in stats_sorted:
        features_str = ','.join(s['features'])
        if len(features_str) > 33:
            features_str = features_str[:30] + '...'
        
        l2re_str = f"{s['l2re_mean']:.2e} ± {s['l2re_std']:.2e}"
        theta_str = f"{s['theta_mean']:.3f}"
        marker = " ★" if s['name'] == best_name else ""
        
        print(f"{s['name']:<15} {features_str:<35} {s['n_inputs']:<5} {s['n_params']:<6} "
              f"{l2re_str:<22} {theta_str:<14} {s['time_mean']:<8.1f}{marker}")
    
    print(f"{'='*110}")
    
    best = stats_sorted[0]
    print(f"\n✓ BEST: {best['name']}")
    print(f"  Features: {best['features']}")
    print(f"  L2RE: {best['l2re_mean']:.2e} ± {best['l2re_std']:.2e}")
    
    # Comparison with default
    default_stat = next((s for s in stats if s['name'] == 'default'), None)
    if default_stat and best['name'] != 'default':
        improvement = (default_stat['l2re_mean'] - best['l2re_mean']) / default_stat['l2re_mean'] * 100
        print(f"  vs default: {improvement:+.1f}%")


# ============================================================
# CUSTOM TRAINING LOOP (needed for adaptive weighting)
# ============================================================

def train_with_weighting(model, domain: dict, epochs: int = 500,
                         n_collocation: int = 30, max_time: float = None,
                         verbose: bool = False):
    """
    Training loop that properly handles adaptive weighting.
    Calls model.step_epoch() after each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    
    history = {
        'losses': [], 'pde_losses': [], 'bc_losses': [], 'ic_losses': [],
        'lambda_pde': [], 'lambda_bc': [], 'lambda_ic': [],
        'theta_statistics': [],
        'epochs_completed': 0,
        'converged': False
    }
    
    model.reset_state()
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss, loss_dict = model.total_loss(domain, n_collocation)
        
        if torch.isnan(total_loss):
            if verbose:
                print(f"   NaN at epoch {epoch}")
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update epoch counter for adaptive strategies
        model.step_epoch()
        
        # Record history
        history['losses'].append(loss_dict['total'])
        history['pde_losses'].append(loss_dict['pde'])
        history['bc_losses'].append(loss_dict['bc'])
        history['ic_losses'].append(loss_dict['ic'])
        history['lambda_pde'].append(loss_dict.get('lambda_pde', model.lambda_pde_base))
        history['lambda_bc'].append(loss_dict.get('lambda_bc', model.lambda_bc_base))
        history['lambda_ic'].append(loss_dict.get('lambda_ic', model.lambda_ic_base))
        
        if hasattr(model, 'get_theta_statistics'):
            history['theta_statistics'].append(model.get_theta_statistics(domain))
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            theta_stats = model.get_theta_statistics(domain)
            print(f"   Epoch {epoch:4d}: Loss={loss_dict['total']:.2e}, "
                  f"PDE={loss_dict['pde']:.2e}, λ=[{loss_dict.get('lambda_pde', 0):.1f}, "
                  f"{loss_dict.get('lambda_bc', 0):.1f}, {loss_dict.get('lambda_ic', 0):.1f}], "
                  f"θ={theta_stats['mean']:.3f}")
        
        # Time limit
        if max_time and time.time() - start_time > max_time:
            break
    
    history['epochs_completed'] = epoch + 1
    history['training_time'] = time.time() - start_time
    
    return history


# ============================================================
# MAIN TEST FUNCTION
# ============================================================

def test_loss_weighting_single_pde(pde_type: str,
                                    strategies: List[str] = None,
                                    n_runs: int = 5,
                                    epochs: int = 500,
                                    max_time: float = None,
                                    # Model parameters
                                    n_steps: int = None,
                                    n_iterations: int = None,
                                    lr: float = None,
                                    n_collocation: int = 30,
                                    # Base weights
                                    lambda_pde: float = 1.0,
                                    lambda_bc: float = 10.0,
                                    lambda_ic: float = 10.0,
                                    verbose: bool = True) -> Tuple[Dict, dict, str]:
    """
    Test different loss weighting strategies for one PDE.
    
    Args:
        pde_type: PDE type ('heat', 'wave', 'burgers', 'reaction_diffusion')
        strategies: list of strategies to test (None = test all)
        n_runs: number of runs per strategy
        epochs: max number of epochs
        max_time: max training time (seconds)
        n_steps, n_iterations, lr: model parameters (None = from PDE_CONFIGS)
        n_collocation: number of collocation points
        lambda_pde, lambda_bc, lambda_ic: base weights
        verbose: print progress
    
    Returns:
        all_results: dict {strategy: [run_results]}
        domain: domain dict
        title: PDE title
    """
    if strategies is None:
        strategies = ['fixed', 'gradual', 'inverse', 'softadapt', 
                      'relobralo', 'ntk', 'self_adaptive', 'causal']
    
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    
    # Get defaults from config
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta':0.5})
    n_steps = n_steps if n_steps is not None else config.get('n_steps', 5)
    n_iterations = n_iterations if n_iterations is not None else config.get('n_iterations', 2)
    lr = lr if lr is not None else config.get('lr', 0.001)
    
    print(f"\n{'='*80}")
    print(f"LOSS WEIGHTING STRATEGY TESTING: {title}")
    print(f"Strategies: {strategies}")
    print(f"Runs per strategy: {n_runs}")
    print(f"Base weights: λ_pde={lambda_pde}, λ_bc={lambda_bc}, λ_ic={lambda_ic}")
    print(f"Model: n_steps={n_steps}, n_iterations={n_iterations}, lr={lr}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy.upper()}")
        print(f"{'='*60}")
        
        strategy_results = []
        
        for run_idx in range(n_runs):
            seed = 42 + run_idx * 111
            if verbose:
                print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Create model with specified strategy
            model = LFPinn_LossWeighting_Test(
                pde_type,
                weighting_strategy=strategy,
                lambda_pde=lambda_pde,
                lambda_bc=lambda_bc,
                lambda_ic=lambda_ic,
                n_steps=config['n_steps'],
                n_iterations=config['n_iterations'],
                lr=config['lr'],
                initial_theta=config['theta']
            )
            
            # Train with custom loop
            history = train_with_weighting(
                model, domain,
                epochs=epochs,
                n_collocation=n_collocation,
                max_time=max_time,
                verbose=verbose
            )
            
            # Evaluate
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['strategy'] = strategy
            
            # Theta statistics
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
            
            # Final weights (for adaptive strategies)
            if history['lambda_pde']:
                res['final_lambda_pde'] = history['lambda_pde'][-1]
                res['final_lambda_bc'] = history['lambda_bc'][-1]
                res['final_lambda_ic'] = history['lambda_ic'][-1]
            
            strategy_results.append(res)
            
            if verbose:
                print(f"    ✓ L2RE={res['l2re']:.4e}, RMSE={res['rmse']:.4e}, "
                      f"θ={theta_stats['mean']:.3f}, Time={res['training_time']:.1f}s")
        
        all_results[strategy] = strategy_results
    
    # Print summary
    if verbose:
        print_loss_weighting_summary(all_results, strategies, pde_type)
    
    return all_results, domain, title


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_loss_weighting_summary(results: Dict, strategies: List[str], pde_type: str):
    """Print summary table"""
    
    print(f"\n{'='*100}")
    print(f"SUMMARY: {pde_type.upper()} - Loss Weighting Strategy Study")
    print(f"{'='*100}")
    print(f"{'Strategy':<15} {'L2RE':<24} {'RMSE':<24} {'θ final':<12} {'Time (s)':<10}")
    print(f"{'-'*100}")
    
    # Collect stats
    stats = []
    for strategy in strategies:
        runs = results[strategy]
        l2res = [r['l2re'] for r in runs]
        rmses = [r['rmse'] for r in runs]
        thetas = [r['theta_final_mean'] for r in runs]
        times = [r['training_time'] for r in runs]
        
        stats.append({
            'strategy': strategy,
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'theta_mean': np.mean(thetas),
            'time_mean': np.mean(times)
        })
    
    # Sort by L2RE
    stats_sorted = sorted(stats, key=lambda x: x['l2re_mean'])
    best_strategy = stats_sorted[0]['strategy']
    
    for s in stats_sorted:
        l2re_str = f"{s['l2re_mean']:.2e} ± {s['l2re_std']:.2e}"
        rmse_str = f"{s['rmse_mean']:.2e} ± {s['rmse_std']:.2e}"
        marker = " ★ BEST" if s['strategy'] == best_strategy else ""
        
        print(f"{s['strategy']:<15} {l2re_str:<24} {rmse_str:<24} "
              f"{s['theta_mean']:<12.3f} {s['time_mean']:<10.1f}{marker}")
    
    print(f"{'='*100}")
    
    # Comparison with fixed baseline
    fixed_stat = next((s for s in stats if s['strategy'] == 'fixed'), None)
    best_stat = stats_sorted[0]
    
    print(f"\n✓ BEST STRATEGY: {best_strategy}")
    print(f"  L2RE: {best_stat['l2re_mean']:.2e} ± {best_stat['l2re_std']:.2e}")
    
    if fixed_stat and best_strategy != 'fixed':
        improvement = (fixed_stat['l2re_mean'] - best_stat['l2re_mean']) / fixed_stat['l2re_mean'] * 100
        print(f"  vs fixed: {improvement:+.1f}%")


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_lf_pinn_with_data_sequential(model, 
                                        domain: dict,
                                        data_points: torch.Tensor,
                                        data_values: torch.Tensor,
                                        pretrain_epochs: int = 300,
                                        finetune_epochs: int = 200,
                                        lf_pretrain_lr: float = 0.05,
                                        finetune_lr: float = 0.0001,
                                        n_collocation: int = 30,
                                        freeze_base: bool = True,
                                        pretrain_max_time: float = None,
                                        finetune_max_time: float = None,
                                        verbose: bool = True) -> dict:
    """
    Train LF-PINN with sequential approach: PDE pretrain → data finetune.
    
    Поддерживает:
      - LowFidelityPINN (без коррекций) — finetune всей модели на данных
      - LFPinn_ThetaParams (с коррекциями) — pretrain без коррекций, finetune с коррекциями
    
    Stage 1: Train on PDE (corrections OFF)
    Stage 2: Finetune on data only (corrections ON)
    
    Для correction_mode='none': finetune обучает всю theta_net на данных.
    Для остальных: finetune обучает correction params (theta_net frozen при freeze_base=True).
    """
    device = next(model.parameters()).device
    data_points = data_points.to(device)
    data_values = data_values.to(device)
    
    has_corrections = hasattr(model, 'correction_mode')
    correction_mode = model.correction_mode if has_corrections else 'none'
    
    # ============================================================
    # STAGE 1: PDE Pretraining (corrections OFF)
    # ============================================================
    if verbose:
        print("    Stage 1: PDE pretraining...")
    
    if has_corrections:
        model.disable_corrections()
        model.freeze_corrections()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lf_pretrain_lr
    )
    
    pretrain_losses = []
    pretrain_pde_losses = []
    pretrain_theta_stats = []
    start_time = time.time()
    
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        loss, loss_dict = model.total_loss(domain, n_collocation)
        
        if torch.isnan(loss):
            break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pretrain_losses.append(loss.item())
        pretrain_pde_losses.append(loss_dict['pde'])
        
        if hasattr(model, 'get_theta_statistics'):
            pretrain_theta_stats.append(model.get_theta_statistics(domain))
        
        # Time limit
        if pretrain_max_time and (time.time() - start_time) > pretrain_max_time:
            break
    
    pretrain_time = time.time() - start_time
    
    if verbose:
        print(f"    Pretrain done: {len(pretrain_losses)} epochs, {pretrain_time:.1f}s")
    
    # ============================================================
    # STAGE 2: Data Finetuning (corrections ON)
    # ============================================================
    if verbose:
        mode_str = f" (mode={correction_mode})" if has_corrections else ""
        print(f"    Stage 2: Data finetuning{mode_str}...")
    
    if has_corrections:
        model.enable_corrections()
        model.unfreeze_corrections()
        
        if correction_mode == 'none':
            # Baseline: дообучаем всю theta_net на данных
            finetune_params = list(model.parameters())
        elif freeze_base:
            model.freeze_base_theta()
            finetune_params = model.get_correction_params()
        else:
            finetune_params = list(model.parameters())
        
        trainable = [p for p in finetune_params if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=finetune_lr)
    else:
        # Обычная LowFidelityPINN — обучаем всё
        optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
    
    finetune_losses = []
    finetune_data_losses = []
    start_time = time.time()
    
    for epoch in range(1, finetune_epochs + 1):
        optimizer.zero_grad()
        
        # Data loss only
        x_data = data_points[:, 0:1]
        t_data = data_points[:, 1:2]
        u_pred = model.forward(x_data, t_data)
        data_loss = torch.mean((u_pred - data_values)**2)
        
        if torch.isnan(data_loss):
            break
        
        data_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        finetune_losses.append(data_loss.item())
        finetune_data_losses.append(data_loss.item())
        
        # Time limit
        if finetune_max_time and (time.time() - start_time) > finetune_max_time:
            break
    
    finetune_time = time.time() - start_time
    
    if verbose:
        print(f"    Finetune done: {len(finetune_losses)} epochs, {finetune_time:.1f}s")
    
    # Combined history
    history = {
        'losses': pretrain_losses + finetune_losses,
        'pde_losses': pretrain_pde_losses + [0] * len(finetune_losses),
        'pretrain_losses': pretrain_losses,
        'finetune_losses': finetune_losses,
        'finetune_data_losses': finetune_data_losses,
        'training_time': pretrain_time + finetune_time,
        'pretrain_time': pretrain_time,
        'finetune_time': finetune_time,
        'pretrain_epochs': len(pretrain_losses),
        'finetune_epochs': len(finetune_losses),
        'epochs_completed': len(pretrain_losses) + len(finetune_losses),
        'theta_statistics': pretrain_theta_stats,
        'correction_mode': correction_mode,
    }
    
    return history


def train_classical_with_data_simultaneous(model: ClassicalPINN,
                                            domain: dict,
                                            data_points: torch.Tensor,
                                            data_values: torch.Tensor,
                                            epochs: int = 500,
                                            lr: float = 0.001,
                                            n_collocation: int = 30,
                                            lambda_pde: float = 1.0,
                                            lambda_bc: float = 10.0,
                                            lambda_data: float = 10.0,
                                            max_time: float = None,
                                            verbose: bool = True) -> dict:
    """
    Train Classical PINN with simultaneous PDE + data training.
    """
    device = next(model.parameters()).device
    data_points = data_points.to(device)
    data_values = data_values.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'losses': [],
        'pde_losses': [],
        'data_losses': [],
        'epochs_completed': 0,
        'training_time': 0
    }
    
    x_min, x_max = domain['x']
    t_min, t_max = domain['t']
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss
        x_col = torch.rand(n_collocation, 1, device=device) * (x_max - x_min) + x_min
        t_col = torch.rand(n_collocation, 1, device=device) * (t_max - t_min) + t_min
        points_col = torch.cat([x_col, t_col], dim=1)
        
        pde_loss = model.pde_loss(points_col)
        bc_loss = model.boundary_loss(domain)
        
        # Data loss
        u_pred = model(data_points)
        data_loss = torch.mean((u_pred - data_values)**2)
        
        total_loss = lambda_pde * pde_loss + lambda_bc * bc_loss + lambda_data * data_loss
        
        if torch.isnan(total_loss):
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['losses'].append(total_loss.item())
        history['pde_losses'].append(pde_loss.item())
        history['data_losses'].append(data_loss.item())
        
        # Time limit
        if max_time and (time.time() - start_time) > max_time:
            break
    
    history['epochs_completed'] = len(history['losses'])
    history['training_time'] = time.time() - start_time
    
    if verbose:
        print(f"    Classical done: {history['epochs_completed']} epochs, {history['training_time']:.1f}s")
    
    return history


# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def run_model_comparison(pde_type: str,
                          n_runs: int = 5,
                          # Correction modes for LF-PINN with data
                          correction_modes: List[str] = None,
                          freeze_base: bool = True,
                          n_time_anchors: int = 5,
                          # LF-PINN settings
                          lf_pretrain_epochs: int = 200,
                          lf_finetune_epochs: int = 1000,
                          lf_finetune_lr: float = 0.0001,
                          # Classical PINN settings
                          classical_epochs: int = 500,
                          classical_lr: float = 0.005,
                          classical_hidden_dims: List[int] = None,
                          # Time limits (optional, overrides epochs)
                          lf_pretrain_max_time: float = None,
                          lf_finetune_max_time: float = None,
                          classical_max_time: float = None,
                          # Data settings
                          n_data_points: int = 50,
                          noise_level: float = 0.05,
                          sampling: str = 'random',
                          # Other
                          n_collocation: int = 30,
                          verbose: bool = True) -> Tuple[Dict, dict, str]:
    """
    Run model comparison study.
    
    Compares:
    - LF-PINN (no data): PDE training only
    - LF-PINN (with data, mode=X): for each correction_mode in correction_modes
    - Classical PINN (with data): simultaneous PDE + data
    
    Args:
        correction_modes: list of correction modes to compare.
            Default: ['none'] (baseline — finetune theta_net on data)
            Example: ['none', 'per_step_bias', 'rhs_scale', 'time_gate+rhs_scale']
            'none' = дообучаем theta_net на данных (как раньше)
        freeze_base: заморозить theta_net при finetune (для mode != 'none')
        n_time_anchors: число якорей для time-based режимов
    """
        
    if classical_hidden_dims is None:
        classical_hidden_dims = [2, 4, 6]
    if correction_modes is None:
        correction_modes = ['none']
    
    domain = DOMAINS[pde_type]
    title = TITLES[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: {title}")
    print(f"{'='*80}")
    print(f"Runs: {n_runs}")
    print(f"LF-PINN: pretrain={lf_pretrain_epochs}ep, finetune={lf_finetune_epochs}ep")
    print(f"Correction modes: {correction_modes}")
    print(f"Classical: {classical_epochs}ep, hidden_dims={classical_hidden_dims}")
    print(f"Data: {n_data_points} points, noise={noise_level}")
    print(f"{'='*80}\n")
    
    # Model names
    model_names = ['LF-PINN (no data)']
    for mode in correction_modes:
        if mode == 'none':
            model_names.append('LF-PINN (with data)')
        else:
            model_names.append(f'LF-PINN ({mode})')
    for h in classical_hidden_dims:
        model_names.append(f'Classical (h={h})')
    
    all_results = {name: [] for name in model_names}
    
    for run_idx in range(n_runs):
        seed = 42 + run_idx * 111
        
        print(f"\n{'='*60}")
        print(f"RUN {run_idx+1}/{n_runs} (seed={seed})")
        print(f"{'='*60}")
        
        # Generate data for this run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain,
            n_points=n_data_points,
            noise_level=noise_level,
            sampling=sampling,
            seed=seed
        )
        
        # ============================================================
        # 1. LF-PINN (no data) - PDE training only
        # ============================================================
        print(f"\n  Training: LF-PINN (no data)...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = LowFidelityPINN(
            pde_type,
            n_steps=config['n_steps'],
            n_iterations=config['n_iterations'],
            lr=config['lr'],
            initial_theta=config['theta'],
            theta_hidden_dim=2
        )
        n_params = sum(p.numel() for p in model.parameters())
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        losses_no_data = []
        start_time = time.time()
        
        for epoch in range(lf_pretrain_epochs):
            optimizer.zero_grad()
            loss, _ = model.total_loss(domain, n_collocation)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses_no_data.append(loss.item())
            
            if lf_pretrain_max_time and (time.time() - start_time) > lf_pretrain_max_time:
                break
        
        training_time = time.time() - start_time
        
        res = evaluate(model, domain, exact_solution=exact_sol)
        history = {
            'losses': losses_no_data,
            'pde_losses': losses_no_data,
            'training_time': training_time,
            'epochs_completed': len(losses_no_data)
        }
        res['history'] = history
        res['training_time'] = training_time
        res['epochs_completed'] = len(losses_no_data)
        res['n_params'] = n_params
        res['model_type'] = 'lf_pinn'
        res['use_data'] = False
        
        if hasattr(model, 'get_theta_statistics'):
            theta_stats = model.get_theta_statistics(domain)
            res['theta_final_mean'] = theta_stats['mean']
            res['theta_final_std'] = theta_stats['std']
        
        all_results['LF-PINN (no data)'].append(res)
        print(f"    ✓ L2RE={res['l2re']:.4e}, Time={res['training_time']:.1f}s")
        
        # ============================================================
        # 2. LF-PINN (with data) - for each correction_mode
        # ============================================================
        for mode in correction_modes:
            if mode == 'none':
                model_name = 'LF-PINN (with data)'
            else:
                model_name = f'LF-PINN ({mode})'
            
            print(f"\n  Training: {model_name}...")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            model = LFPinn_ThetaParams(
                pde_type,
                correction_mode=mode,
                n_steps=config['n_steps'],
                theta_hidden_dim=2,
                n_iterations=config['n_iterations'],
                initial_theta=config['theta'],
                lr=config['lr'],
                n_time_anchors=n_time_anchors,
                t_max=domain['t'][1]
            )
            n_params = sum(p.numel() for p in model.parameters())
            
            history = train_lf_pinn_with_data_sequential(
                model, domain, data_points, data_values,
                pretrain_epochs=lf_pretrain_epochs,
                finetune_epochs=lf_finetune_epochs,
                lf_pretrain_lr=config['lr'],
                finetune_lr=lf_finetune_lr,
                n_collocation=n_collocation,
                freeze_base=freeze_base,
                pretrain_max_time=lf_pretrain_max_time,
                finetune_max_time=lf_finetune_max_time,
                verbose=verbose
            )
            
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['n_params'] = n_params
            res['model_type'] = 'lf_pinn'
            res['use_data'] = True
            res['correction_mode'] = mode
            res['pretrain_time'] = history['pretrain_time']
            res['finetune_time'] = history['finetune_time']
            
            if hasattr(model, 'get_theta_statistics'):
                theta_stats = model.get_theta_statistics(domain)
                res['theta_final_mean'] = theta_stats['mean']
                res['theta_final_std'] = theta_stats['std']
            
            all_results[model_name].append(res)
            print(f"    ✓ L2RE={res['l2re']:.4e}, Time={res['training_time']:.1f}s")
        
        # ============================================================
        # 3. Classical PINNs (with data, simultaneous)
        # ============================================================
        for hidden_dim in classical_hidden_dims:
            name = f'Classical (h={hidden_dim})'
            print(f"\n  Training: {name}...")
            
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            model = ClassicalPINN(pde_type, hidden_dim=hidden_dim)
            n_params = sum(p.numel() for p in model.parameters())
            
            history = train_classical_with_data_simultaneous(
                model, domain, data_points, data_values,
                epochs=classical_epochs,
                lr=classical_lr,
                n_collocation=n_collocation,
                max_time=classical_max_time,
                verbose=verbose
            )
            
            res = evaluate(model, domain, exact_solution=exact_sol)
            res['history'] = history
            res['training_time'] = history['training_time']
            res['epochs_completed'] = history['epochs_completed']
            res['n_params'] = n_params
            res['model_type'] = 'classical'
            res['use_data'] = True
            res['hidden_dim'] = hidden_dim
            
            all_results[name].append(res)
            print(f"    ✓ L2RE={res['l2re']:.4e}, Params={n_params}, Time={res['training_time']:.1f}s")
    
    # Print summary
    print_comparison_summary(all_results, pde_type)
    
    return all_results, domain, title

# ============================================================
# SUMMARY TABLE
# ============================================================

def print_comparison_summary(results: Dict, pde_type: str):
    """Print comparison summary table."""
    print(f"\n{'='*120}")
    print(f"SUMMARY: {pde_type.upper()} Model Comparison")
    print(f"{'='*120}")
    print(f"{'Model':<30} {'Type':<10} {'Data':<6} {'Params':<8} {'Corr.P':<8} {'L2RE':<24} {'RMSE':<24} {'Time (s)':<12}")
    print(f"{'-'*120}")
    
    stats = []
    for name, runs in results.items():
        if not runs:
            continue
        
        l2res = [r['l2re'] for r in runs]
        rmses = [r['rmse'] for r in runs]
        times = [r['training_time'] for r in runs]
        
        stats.append({
            'name': name,
            'type': runs[0].get('model_type', 'unknown'),
            'use_data': runs[0].get('use_data', False),
            'n_params': runs[0].get('n_params', 0),
            'n_correction_params': runs[0].get('n_correction_params', 0),
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'rmse_mean': np.mean(rmses),
            'rmse_std': np.std(rmses),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
        })
    
    stats_sorted = sorted(stats, key=lambda x: x['l2re_mean'])
    best_name = stats_sorted[0]['name']
    
    for s in stats_sorted:
        data_str = "Yes" if s['use_data'] else "No"
        l2re_str = f"{s['l2re_mean']:.2e} ± {s['l2re_std']:.2e}"
        rmse_str = f"{s['rmse_mean']:.2e} ± {s['rmse_std']:.2e}"
        time_str = f"{s['time_mean']:.1f} ± {s['time_std']:.1f}"
        corr_str = str(s['n_correction_params']) if s['n_correction_params'] else '--'
        marker = " ★" if s['name'] == best_name else ""
        
        print(f"{s['name']:<30} {s['type']:<10} {data_str:<6} {s['n_params']:<8} "
              f"{corr_str:<8} {l2re_str:<24} {rmse_str:<24} {time_str:<12}{marker}")
    
    print(f"{'='*120}")
    
    best = stats_sorted[0]
    print(f"\n✓ BEST: {best['name']}")
    print(f"  L2RE: {best['l2re_mean']:.2e} ± {best['l2re_std']:.2e}")
    print(f"  Time: {best['time_mean']:.1f}s")


def pde_training_l2re_study(pde_type: str,
                             n_runs: int = 5,
                             max_epochs: int = 1000,
                             eval_every: int = 10,
                             n_collocation: int = 30,
                             lr: float = None,
                             verbose: bool = True):
    """
    Study L2RE evolution during PDE-only training.
    
    Tracks L2RE every eval_every epochs to find:
    - When L2RE drops the most
    - Optimal stopping point
    - Convergence behavior
    
    Args:
        pde_type: 'heat', 'wave', 'burgers', 'reaction_diffusion'
        n_runs: number of runs
        max_epochs: maximum training epochs
        eval_every: evaluate L2RE every N epochs
        n_collocation: collocation points
        lr: learning rate (None = use config default)
        verbose: print progress
    
    Returns:
        dict with detailed results
    """
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS.get(pde_type, {'n_steps': 5, 'n_iterations': 2, 'lr': 0.001, 'theta': 0.5})
    
    if lr is None:
        lr = config['lr']
    
    print(f"\n{'='*70}")
    print(f"PDE TRAINING L2RE STUDY: {TITLES[pde_type]}")
    print(f"{'='*70}")
    print(f"Runs: {n_runs}, Max epochs: {max_epochs}, Eval every: {eval_every}")
    print(f"LR: {lr}, n_collocation: {n_collocation}")
    print(f"{'='*70}\n")
    
    # Storage for all runs
    all_epochs = []
    all_l2re = []
    all_rmse = []
    all_loss = []
    all_theta = []
    
    for run in range(n_runs):
        if verbose:
            print(f"\nRun {run+1}/{n_runs}")
        
        seed = 42 + 111 * run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create model
        model = LowFidelityPINN(
            pde_type,
            n_steps=config['n_steps'],
            n_iterations=config['n_iterations'],
            lr=lr,
            initial_theta=config['theta']
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Track metrics
        run_epochs = [0]
        run_l2re = []
        run_rmse = []
        run_loss = [float('nan')]
        run_theta = []
        
        # Initial evaluation (epoch 0)
        res = evaluate(model, domain, exact_solution=exact_sol)
        run_l2re.append(res['l2re'])
        run_rmse.append(res['rmse'])
        run_theta.append(model.get_theta_statistics(domain)['mean'])
        
        if verbose:
            print(f"  Epoch 0: L2RE={res['l2re']:.2e}")
        
        # Training loop
        for epoch in range(1, max_epochs + 1):
            optimizer.zero_grad()
            loss, loss_dict = model.total_loss(domain, n_collocation)
            
            if torch.isnan(loss):
                print(f"  NaN at epoch {epoch}, stopping run")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Evaluate every eval_every epochs
            if epoch % eval_every == 0:
                res = evaluate(model, domain, exact_solution=exact_sol)
                theta_stats = model.get_theta_statistics(domain)
                
                run_epochs.append(epoch)
                run_l2re.append(res['l2re'])
                run_rmse.append(res['rmse'])
                run_loss.append(loss.item())
                run_theta.append(theta_stats['mean'])
                
                if verbose and epoch % (eval_every * 10) == 0:
                    print(f"  Epoch {epoch}: L2RE={res['l2re']:.2e}, Loss={loss.item():.2e}, θ={theta_stats['mean']:.3f}")
        
        all_epochs.append(run_epochs)
        all_l2re.append(run_l2re)
        all_rmse.append(run_rmse)
        all_loss.append(run_loss)
        all_theta.append(run_theta)
        
        if verbose:
            print(f"  Final: L2RE={run_l2re[-1]:.2e}, θ={run_theta[-1]:.3f}")
    
    # ============================================================
    # Aggregate results
    # ============================================================
    max_len = max(len(e) for e in all_epochs)
    epochs_common = all_epochs[0][:max_len]
    
    # Pad arrays to same length
    def pad_array(arr_list, max_len):
        return np.array([arr + [np.nan]*(max_len - len(arr)) for arr in arr_list])
    
    l2re_matrix = pad_array(all_l2re, max_len)
    rmse_matrix = pad_array(all_rmse, max_len)
    loss_matrix = pad_array(all_loss, max_len)
    theta_matrix = pad_array(all_theta, max_len)
    
    l2re_mean = np.nanmean(l2re_matrix, axis=0)
    l2re_std = np.nanstd(l2re_matrix, axis=0)
    l2re_min = np.nanmin(l2re_matrix, axis=0)
    l2re_max = np.nanmax(l2re_matrix, axis=0)
    
    rmse_mean = np.nanmean(rmse_matrix, axis=0)
    loss_mean = np.nanmean(loss_matrix, axis=0)
    theta_mean = np.nanmean(theta_matrix, axis=0)
    theta_std = np.nanstd(theta_matrix, axis=0)
    
    # ============================================================
    # Find steepest L2RE drop
    # ============================================================
    # Compute L2RE drop rate (negative = improvement)
    l2re_drop = np.diff(l2re_mean)  # l2re[i+1] - l2re[i]
    l2re_drop_rate = l2re_drop / eval_every  # per epoch
    
    # Relative drop (percentage improvement per eval_every epochs)
    l2re_relative_drop = l2re_drop / (l2re_mean[:-1] + 1e-10) * 100
    
    # Find best drop indices
    best_drop_idx = np.nanargmin(l2re_drop)  # Most negative = best improvement
    best_relative_drop_idx = np.nanargmin(l2re_relative_drop)
    
    best_drop_epoch = epochs_common[best_drop_idx + 1]
    best_relative_drop_epoch = epochs_common[best_relative_drop_idx + 1]
    
    # Find minimum L2RE
    min_l2re_idx = np.nanargmin(l2re_mean)
    min_l2re_epoch = epochs_common[min_l2re_idx]
    min_l2re_value = l2re_mean[min_l2re_idx]
    
    # Find 90%, 95%, 99% of best improvement
    initial_l2re = l2re_mean[0]
    final_l2re = l2re_mean[-1]
    improvement = initial_l2re - final_l2re
    
    thresholds = {'90%': 0.90, '95%': 0.95, '99%': 0.99}
    threshold_epochs = {}
    for name, pct in thresholds.items():
        target = initial_l2re - pct * improvement
        idx = np.where(l2re_mean <= target)[0]
        if len(idx) > 0:
            threshold_epochs[name] = epochs_common[idx[0]]
        else:
            threshold_epochs[name] = None
    
    # ============================================================
    # Print analysis
    # ============================================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {pde_type.upper()}")
    print(f"{'='*70}")
    print(f"\nL2RE Evolution:")
    print(f"  Initial (epoch 0):     {initial_l2re:.2e}")
    print(f"  Final (epoch {max_epochs}):    {final_l2re:.2e}")
    print(f"  Minimum:               {min_l2re_value:.2e} at epoch {min_l2re_epoch}")
    print(f"  Total improvement:     {(initial_l2re - final_l2re) / initial_l2re * 100:.1f}%")
    
    print(f"\nSteepest L2RE drop:")
    print(f"  Absolute: epoch {best_drop_epoch} (drop = {l2re_drop[best_drop_idx]:.2e})")
    print(f"  Relative: epoch {best_relative_drop_epoch} (drop = {l2re_relative_drop[best_relative_drop_idx]:.1f}%)")
    
    print(f"\nConvergence milestones:")
    for name, ep in threshold_epochs.items():
        if ep is not None:
            print(f"  {name} of improvement: epoch {ep}")
        else:
            print(f"  {name} of improvement: not reached")
    
    print(f"\nTheta evolution:")
    print(f"  Initial: {theta_mean[0]:.3f}")
    print(f"  Final:   {theta_mean[-1]:.3f} ± {theta_std[-1]:.3f}")
    
    print(f"{'='*70}")
    
    results = {
        'pde_type': pde_type,
        'epochs': epochs_common,
        'l2re_mean': l2re_mean,
        'l2re_std': l2re_std,
        'l2re_min': l2re_min,
        'l2re_max': l2re_max,
        'rmse_mean': rmse_mean,
        'loss_mean': loss_mean,
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'l2re_drop': l2re_drop,
        'l2re_relative_drop': l2re_relative_drop,
        'best_drop_epoch': best_drop_epoch,
        'best_relative_drop_epoch': best_relative_drop_epoch,
        'min_l2re_epoch': min_l2re_epoch,
        'min_l2re_value': min_l2re_value,
        'threshold_epochs': threshold_epochs,
        'all_l2re': all_l2re,
        'all_epochs': all_epochs,
        'all_theta': all_theta,
        'n_runs': n_runs,
        'eval_every': eval_every,
    }
    
    return results


def test_theta_param_modes(pde_type: str = 'heat',
                            n_runs: int = 3,
                            pretrain_epochs: int = 200,
                            finetune_epochs: int = 500,
                            pretrain_lr: float = 0.005,
                            finetune_lr: float = 0.001,
                            n_data: int = 50,
                            noise_level: float = 0.05,
                            freeze_base: bool = True):
    """
    Сравнение режимов параметрической коррекции theta (без изменения n_steps).
    """
    
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS[pde_type]
    
    modes = ['none', 'per_step_bias', 'per_step_gate', 'time_bias', 'time_gate', 'rhs_scale', 'output_bias', 'step_bias+output_bias', 'time_bias+output_bias', 'time_gate+rhs_scale']
    
    results = {mode: [] for mode in modes}
    
    for run in range(n_runs):
        seed = 42 + run * 111
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain, n_points=n_data,
            noise_level=noise_level, seed=seed
        )
        
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{n_runs} (seed={seed})")
        print(f"{'='*70}")
        
        for mode in modes:
            torch.manual_seed(seed)
            
            model = LFPinn_ThetaParams(
                pde_type,
                correction_mode=mode,
                n_steps=config['n_steps'],
                theta_hidden_dim=2,
                n_iterations=config['n_iterations'],
                initial_theta=config['theta'],
                lr=config['lr'],
                t_max=domain['t'][1]
            )
            
            result = _run_pretrain_finetune_thetaparams(
                model, domain, exact_sol, data_points, data_values,
                pretrain_epochs, finetune_epochs, pretrain_lr, finetune_lr,
                freeze_base
            )
            results[mode].append(result)
            
            print(f"  {mode:20s}: pretrain={result['l2re_pretrain']:.4e} → finetune={result['l2re_finetune']:.4e} "
                  f"(Δ={result['improvement_pct']:+.1f}%, corr_params={result['n_correction_params']})")
    
    _print_summary("PARAM MODES", pde_type, modes, results, n_runs, freeze_base)
    return results


# ============================================================
# ТЕСТ 2: Увеличение n_steps при finetune
# ============================================================

def test_step_scaling(pde_type: str = 'heat',
                       n_runs: int = 3,
                       pretrain_epochs: int = 200,
                       finetune_epochs: int = 500,
                       pretrain_lr: float = 0.005,
                       finetune_lr: float = 0.001,
                       n_data: int = 50,
                       noise_level: float = 0.05,
                       freeze_base: bool = True,
                       finetune_steps_list: list = None):
    """
    Pretrain с n_steps=K (из конфига), finetune с n_steps из списка.
    
    Идея: PDE loss с большим n_steps не влезает в память (вложенный autograd),
    но data loss (просто forward + MSE) — дешевле, и можно поставить больше шагов.
    
    Args:
        finetune_steps_list: список значений n_steps для finetune, 
                             default берёт base из конфига: [K, K+1, K+2, K+3]
    """
    
    domain = DOMAINS[pde_type]
    exact_sol = get_exact_solution_parametric(pde_type)
    config = PDE_CONFIGS[pde_type]
    base_n_steps = config['n_steps']
    
    if finetune_steps_list is None:
        finetune_steps_list = [base_n_steps, base_n_steps + 1, base_n_steps + 2, base_n_steps + 3]
    
    correction_modes = ['per_step_bias', 'per_step_gate']
    all_modes = ['none'] + correction_modes
    
    # Ключ: (mode, ft_steps)
    results = {}
    for mode in all_modes:
        for ft_steps in finetune_steps_list:
            results[(mode, ft_steps)] = []
    
    for run in range(n_runs):
        seed = 42 + run * 111
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        data_points, data_values, _ = generate_synthetic_data(
            pde_type, domain, n_points=n_data,
            noise_level=noise_level, seed=seed
        )
        
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{n_runs} (seed={seed}) | base n_steps={base_n_steps}")
        print(f"{'='*70}")
        
        for mode in all_modes:
            for ft_steps in finetune_steps_list:
                
                torch.manual_seed(seed)
                
                model = LFPinn_ThetaParams(
                    pde_type,
                    correction_mode=mode,
                    n_steps=base_n_steps,
                    theta_hidden_dim=2,
                    n_iterations=config['n_iterations'],
                    initial_theta=config['theta'],
                    lr=config['lr'],
                    t_max=domain['t'][1]
                )
                
                # ========== STAGE 1: PDE Pretrain (n_steps = base) ==========
                model.disable_corrections()
                model.freeze_corrections()
                
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=pretrain_lr
                )
                
                for epoch in range(pretrain_epochs):
                    optimizer.zero_grad()
                    loss, _ = model.total_loss(domain, n_collocation=30)
                    if torch.isnan(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
                
                # ========== STAGE 2: Увеличиваем n_steps ==========
                if ft_steps != base_n_steps:
                    model.change_n_steps(ft_steps)
                
                # ========== STAGE 3: Data Finetune ==========
                model.enable_corrections()
                model.unfreeze_corrections()
                
                if freeze_base:
                    model.freeze_base_theta()
                    finetune_params = model.get_correction_params()
                else:
                    finetune_params = list(model.parameters())
                
                trainable = [p for p in finetune_params if p.requires_grad]
                
                if len(trainable) == 0:
                    res_finetune = res_pretrain
                else:
                    optimizer = torch.optim.Adam(trainable, lr=finetune_lr)
                    x_data = data_points[:, 0:1]
                    t_data = data_points[:, 1:2]
                    
                    for epoch in range(finetune_epochs):
                        optimizer.zero_grad()
                        data_loss = model.data_loss(x_data, t_data, data_values)
                        if torch.isnan(data_loss):
                            break
                        data_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    res_finetune = evaluate(model, domain, exact_solution=exact_sol)
                
                imp = 0.0
                if res_pretrain['l2re'] > 1e-12:
                    imp = (res_pretrain['l2re'] - res_finetune['l2re']) / res_pretrain['l2re'] * 100
                
                n_corr = model.get_params()['n_correction_params']
                
                results[(mode, ft_steps)].append({
                    'l2re_pretrain': res_pretrain['l2re'],
                    'l2re_finetune': res_finetune['l2re'],
                    'improvement_pct': imp,
                    'n_correction_params': n_corr,
                    'pretrain_n_steps': base_n_steps,
                    'finetune_n_steps': ft_steps,
                    'correction_stats': model.get_correction_statistics(domain),
                })
                
                print(f"  {mode:16s}: steps {base_n_steps}→{ft_steps}, "
                      f"pretrain={res_pretrain['l2re']:.4e} → finetune={res_finetune['l2re']:.4e} "
                      f"(Δ={imp:+.1f}%, params={n_corr})")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*100}")
    print(f"STEP SCALING: {pde_type.upper()} | pretrain n_steps={base_n_steps} | {n_runs} runs")
    print(f"{'='*100}")
    print(f"{'Mode':<18} {'FT steps':<10} {'Pretrain L2RE':<16} {'Finetune L2RE':<22} {'Δ%':<12} {'Params':<8}")
    print("-" * 100)
    
    for mode in all_modes:
        for ft_steps in finetune_steps_list:
            runs = results[(mode, ft_steps)]
            pre = np.mean([r['l2re_pretrain'] for r in runs])
            ft_mean = np.mean([r['l2re_finetune'] for r in runs])
            ft_std = np.std([r['l2re_finetune'] for r in runs])
            imp = np.mean([r['improvement_pct'] for r in runs])
            n_corr = runs[0]['n_correction_params']
            
            print(f"{mode:<18} {ft_steps:<10} {pre:<16.4e} {ft_mean:.4e}±{ft_std:.4e} "
                  f"{imp:>+8.1f}%    {n_corr:<8}")
        print()
    
    return results


# ============================================================
# Вспомогательные функции
# ============================================================

def _run_pretrain_finetune_thetaparams(model, domain, exact_sol,
                                        data_points, data_values,
                                        pretrain_epochs, finetune_epochs,
                                        pretrain_lr, finetune_lr,
                                        freeze_base):
    """Pretrain + finetune для LFPinn_ThetaParams."""
    from utils.pde_utils import evaluate
    
    # Pretrain
    model.disable_corrections()
    model.freeze_corrections()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=pretrain_lr
    )
    
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        loss, _ = model.total_loss(domain, n_collocation=30)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
    
    # Finetune
    model.enable_corrections()
    model.unfreeze_corrections()
    
    if model.correction_mode == 'none':
        # Baseline: дообучаем theta_net на данных (как train_lf_pinn_with_data_sequential)
        finetune_params = list(model.parameters())
    elif freeze_base:
        model.freeze_base_theta()
        finetune_params = model.get_correction_params()
    else:
        finetune_params = list(model.parameters())
    
    trainable = [p for p in finetune_params if p.requires_grad]
    if len(trainable) == 0:
        return {
            'l2re_pretrain': res_pretrain['l2re'],
            'l2re_finetune': res_pretrain['l2re'],
            'improvement_pct': 0.0,
            'n_correction_params': model.get_params()['n_correction_params'],
        }
    
    optimizer = torch.optim.Adam(trainable, lr=finetune_lr)
    x_data = data_points[:, 0:1]
    t_data = data_points[:, 1:2]
    
    for epoch in range(finetune_epochs):
        optimizer.zero_grad()
        loss = model.data_loss(x_data, t_data, data_values)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    res_finetune = evaluate(model, domain, exact_solution=exact_sol)
    
    imp = 0.0
    if res_pretrain['l2re'] > 1e-12:
        imp = (res_pretrain['l2re'] - res_finetune['l2re']) / res_pretrain['l2re'] * 100
    
    return {
        'l2re_pretrain': res_pretrain['l2re'],
        'l2re_finetune': res_finetune['l2re'],
        'improvement_pct': imp,
        'n_correction_params': model.get_params()['n_correction_params'],
        'correction_stats': model.get_correction_statistics(domain),
    }


def _run_pretrain_finetune_datacorr(model, domain, exact_sol,
                                     data_points, data_values,
                                     pretrain_epochs, finetune_epochs,
                                     pretrain_lr, finetune_lr,
                                     freeze_base, mode):
    """Pretrain + finetune для LFPinn_DataCorrection_Test."""
    from utils.pde_utils import evaluate
    
    # Pretrain
    model.disable_corrections()
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
    
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        loss, _ = model.total_loss(domain, n_collocation=30)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    res_pretrain = evaluate(model, domain, exact_solution=exact_sol)
    
    # Finetune
    model.enable_corrections()
    
    if freeze_base:
        model.freeze_base_theta()
        if mode == 'base_only':
            finetune_params = list(model.parameters())
        elif mode == 'shared_corr':
            finetune_params = list(model.correction_net.parameters())
        elif mode == 'per_step_corr':
            finetune_params = list(model.correction_nets.parameters())
        elif mode == 'per_step_scalar':
            finetune_params = [model.correction_scalars]
    else:
        finetune_params = list(model.parameters())
    
    trainable = [p for p in finetune_params if p.requires_grad]
    if len(trainable) == 0:
        return {
            'l2re_pretrain': res_pretrain['l2re'],
            'l2re_finetune': res_pretrain['l2re'],
            'improvement_pct': 0.0,
            'n_correction_params': model.get_params()['n_correction_params'],
        }
    
    optimizer = torch.optim.Adam(trainable, lr=finetune_lr)
    x_data = data_points[:, 0:1]
    t_data = data_points[:, 1:2]
    
    for epoch in range(finetune_epochs):
        optimizer.zero_grad()
        loss = model.data_loss(x_data, t_data, data_values)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    res_finetune = evaluate(model, domain, exact_solution=exact_sol)
    
    imp = 0.0
    if res_pretrain['l2re'] > 1e-12:
        imp = (res_pretrain['l2re'] - res_finetune['l2re']) / res_pretrain['l2re'] * 100
    
    return {
        'l2re_pretrain': res_pretrain['l2re'],
        'l2re_finetune': res_finetune['l2re'],
        'improvement_pct': imp,
        'n_correction_params': model.get_params()['n_correction_params'],
    }


def _print_summary(title, pde_type, modes, results, n_runs, freeze_base):
    """Общая таблица итогов."""
    print(f"\n{'='*90}")
    print(f"{title}: {pde_type.upper()} | {n_runs} runs | freeze_base={freeze_base}")
    print(f"{'='*90}")
    print(f"{'Mode':<20} {'Pretrain L2RE':<16} {'Finetune L2RE':<22} {'Δ%':<14} {'Corr Params':<12}")
    print("-" * 90)
    
    for mode in modes:
        pre = np.mean([r['l2re_pretrain'] for r in results[mode]])
        ft_mean = np.mean([r['l2re_finetune'] for r in results[mode]])
        ft_std = np.std([r['l2re_finetune'] for r in results[mode]])
        imp = np.mean([r['improvement_pct'] for r in results[mode]])
        n_corr = results[mode][0]['n_correction_params']
        
        print(f"{mode:<20} {pre:<16.4e} {ft_mean:.4e}±{ft_std:.4e} "
              f"{imp:>+10.1f}%    {n_corr:<12}")