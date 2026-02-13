import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple
from utils.config import TITLES
import numpy as np
import torch


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def plot(results, history, title="PINN Solution"):
    points = results['points']
    u_pred = results['u_pred'].detach().numpy()
    u_exact = results.get('u_exact', None)
    grid_shape = results['grid_shape']
    is_spatial_2d = results['is_spatial_2d']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. PINN prediction
    ax1 = axes[0, 0]
    if is_spatial_2d:
        x = points[:, 0].numpy().reshape(grid_shape)
        y = points[:, 1].numpy().reshape(grid_shape)
        u = u_pred.reshape(grid_shape)
        im1 = ax1.contourf(x, y, u, levels=50, cmap='viridis')
        ax1.set_xlabel('x'), ax1.set_ylabel('y'), ax1.set_title('PINN u(x,y)')
        plt.colorbar(im1, ax=ax1)
    else:
        ax1.remove()
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        x = points[:, 0].numpy().reshape(grid_shape)
        t = points[:, 1].numpy().reshape(grid_shape)
        u = u_pred.reshape(grid_shape)
        ax1.plot_surface(x, t, u, cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('x'), ax1.set_ylabel('t'), ax1.set_zlabel('u')
        ax1.set_title('PINN u(x,t)'), ax1.view_init(elev=25, azim=45)
    
    # 2. Training history
    ax2 = axes[0, 1]
    epochs = range(len(history['losses']))
    ax2.semilogy(epochs, history['losses'], 'b-', linewidth=2, label='Total Loss', alpha=0.8)
    ax2.semilogy(epochs, history['pde_losses'], 'r-', linewidth=1.5, label='PDE', alpha=0.7)
    
    # Optional: BC and IC if available
    if 'bc_losses' in history and any(bc > 0 for bc in history['bc_losses']):
        ax2.semilogy(epochs, history['bc_losses'], 'g-', linewidth=1.5, label='BC', alpha=0.7)
    if 'ic_losses' in history and any(ic > 0 for ic in history['ic_losses']):
        ax2.semilogy(epochs, history['ic_losses'], 'm-', linewidth=1.5, label='IC', alpha=0.7)
    
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Loss (log)')
    ax2.set_title('Training History'), ax2.legend(), ax2.grid(True, alpha=0.3)
    
    # 3. Exact solution
    ax3 = axes[0, 2]
    if u_exact is not None:
        if is_spatial_2d:
            u_ex = u_exact.detach().numpy().reshape(grid_shape)
            im3 = ax3.contourf(x, y, u_ex, levels=50, cmap='viridis')
            ax3.set_xlabel('x'), ax3.set_ylabel('y'), ax3.set_title('Exact Solution')
            plt.colorbar(im3, ax=ax3)
        else:
            ax3.remove()
            ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            u_ex = u_exact.detach().numpy().reshape(grid_shape)
            ax3.plot_surface(x, t, u_ex, cmap='viridis', alpha=0.8, edgecolor='none')
            ax3.set_xlabel('x'), ax3.set_ylabel('t'), ax3.set_zlabel('u')
            ax3.set_title('Exact Solution'), ax3.view_init(elev=25, azim=45)
    else:
        ax3.text(0.5, 0.5, 'No Exact\nSolution', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14, color='gray')
        ax3.axis('off')
    
    # 4. Theta parameter θ(x,t)
    ax4 = axes[1, 0]
    if 'theta_statistics' in history and history['theta_statistics'] and history['theta_statistics'][0]:
        theta_stats = [s for s in history['theta_statistics'] if s is not None]
        if theta_stats:
            theta_means = [s['mean'] for s in theta_stats]
            theta_stds = [s['std'] for s in theta_stats]
            epochs_theta = range(len(theta_means))
            
            ax4.plot(epochs_theta, theta_means, 'r-', linewidth=2, label='θ (mean)')
            if any(std > 0 for std in theta_stds):
                ax4.fill_between(epochs_theta, 
                                np.array(theta_means) - np.array(theta_stds),
                                np.array(theta_means) + np.array(theta_stds),
                                alpha=0.3, color='red', label='±σ')
            
            ax4.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Trapezoidal')
            ax4.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit')
            ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit')
            
            ax4.set_xlabel('Epoch'), ax4.set_ylabel('θ')
            ax4.set_title('Theta Evolution'), ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3), ax4.set_ylim(-0.05, 1.05)
    else:
        ax4.text(0.5, 0.5, 'No θ\nData', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, color='gray')
        ax4.axis('off')
    
    # 6. Time slices
    ax6 = axes[1, 2]
    if u_exact is not None and not is_spatial_2d:
        x_vals = points[:, 0].numpy().reshape(grid_shape)
        t_vals = points[:, 1].numpy().reshape(grid_shape)
        u_pred_vals = u_pred.reshape(grid_shape)
        u_exact_vals = u_exact.detach().numpy().reshape(grid_shape)
        
        times = np.linspace(t_vals[0, :].min(), t_vals[0, :].max(), 5)
        colors = plt.cm.viridis(np.linspace(0, 1, 5))
        
        for i, t0 in enumerate(times):
            idx = np.argmin(np.abs(t_vals[0, :] - t0))
            ax6.plot(x_vals[:, idx], u_exact_vals[:, idx], color=colors[i], 
                    linestyle='-', linewidth=2.5, alpha=0.7)
            ax6.plot(x_vals[:, idx], u_pred_vals[:, idx], color=colors[i],
                    linestyle='--', linewidth=2, alpha=0.9)
        
        from matplotlib.lines import Line2D
        ax6.legend(handles=[Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Exact'),
                           Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='PINN')],
                  loc='upper right', fontsize=10)
        ax6.set_xlabel('x'), ax6.set_ylabel('u'), ax6.set_title('Time Slices')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if history['params']:
        print(f"\n{'='*60}")
        print("Final Parameters:")
        for key, val in history['params'][-1].items():
            if isinstance(val, (int, float)):
                print(f"  {key:20s} = {val:.6f}")
        print(f"Final Loss: {history['losses'][-1]:.4e}")
        if 'error' in results:
            print(f"Mean Error: {results['mean_error']:.4e}")
            print(f"L2RE Error:  {results['l2re']:.4e}")
        print(f"{'='*60}\n")


# ============================================================
# METHOD COMPARISON
# ============================================================

def plot_comparison(results: dict, title: str):
    methods = ['learnable_theta', 'explicit', 'implicit', 'trapezoidal', 'classical_pinn']
    names = ['Low-Fidelity', 'Explicit', 'Implicit', 'Trapezoidal', 'Classical']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    existing = [(m, n, c) for m, n, c in zip(methods, names, colors) 
                if m in results and results[m] and isinstance(results[m], dict) and 'error' not in results[m]]
    
    if not existing:
        print("No data for visualization")
        return
    
    ex_methods, ex_names, ex_colors = zip(*existing)
    
    # Losses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Training: {title}', fontsize=16, fontweight='bold')
    
    for m, n, c in existing:
        if 'history' in results[m] and 'losses' in results[m]['history']:
            ax1.semilogy(results[m]['history']['losses'], label=n, linewidth=2, color=c)
    ax1.set_xlabel('Epoch'), ax1.set_ylabel('Total Loss')
    ax1.legend(), ax1.grid(True, alpha=0.3)
    
    for m, n, c in existing:
        if 'history' in results[m] and 'pde_losses' in results[m]['history']:
            ax2.semilogy(results[m]['history']['pde_losses'], label=n, linewidth=2, color=c)
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('PDE Residual')
    ax2.legend(), ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# THETA FIELD VISUALIZATION
# ============================================================

def plot_theta_field(model, domain, title="", n_grid=50):
    if not hasattr(model, 'theta_net'):
        print("Model doesn't support theta field visualization")
        return
    
    x_min, x_max = domain['x']
    t_min, t_max = domain['t']
    
    x = torch.linspace(x_min, x_max, n_grid)
    t = torch.linspace(t_min, t_max, n_grid)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    x_flat = X.flatten().unsqueeze(1)
    t_flat = T.flatten().unsqueeze(1)
    h = (t_max - t_min) / model.n_steps
    t_next = torch.clamp(t_flat + h, max=t_max)
    
    with torch.no_grad():
        # Approximation via IC (fast)
        y_approx = model.initial_condition(x_flat)
        if model.is_wave:
            y_approx = y_approx[:, 0:1]
        
        # grad_norm = 0 for visualization
        grad_norm_dummy = torch.zeros_like(x_flat)
        
        # 4 inputs to theta_net
        theta_values = model.theta_net(torch.cat([
            x_flat,
            t_flat,
            t_next,
            grad_norm_dummy
        ], dim=1))
    
    theta_grid = theta_values.reshape(n_grid, n_grid).numpy()
    
    fig = plt.figure(figsize=(16, 6))
    
    # 2D heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.contourf(X.numpy(), T.numpy(), theta_grid, levels=50, cmap='RdYlGn_r')
    ax1.contour(X.numpy(), T.numpy(), theta_grid, levels=[0.0, 0.5, 1.0], 
                colors=['blue', 'green', 'orange'], linewidths=2, alpha=0.7)
    plt.colorbar(im, ax=ax1, label='θ')
    ax1.set_xlabel('x'), ax1.set_ylabel('t')
    ax1.set_title(f'θ(x,t) Field - {title}', fontweight='bold')
    
    # 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X.numpy(), T.numpy(), theta_grid, cmap='RdYlGn_r', alpha=0.9)
    ax2.set_xlabel('x'), ax2.set_ylabel('t'), ax2.set_zlabel('θ')
    ax2.set_title('3D θ(x,t)', fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nθ Statistics: mean={theta_grid.mean():.4f}, std={theta_grid.std():.4f}, "
          f"min={theta_grid.min():.4f}, max={theta_grid.max():.4f}")


# ============================================================
# ACTIVATION FUNCTION COMPARISON
# ============================================================
    
def plot_activation_comparison(all_results, title, domain):
    """
    Visualizes comparison of different activation functions.
    Classic scientific style.
    """
    activations = list(all_results.keys())
    n_acts = len(activations)
    
    # Collect statistics
    stats = {act: {
        'l2res': [], 'rmses': [], 'max_errs': [], 
        'training_times': [], 'pde_residuals': [],
        'theta_means': [], 'epochs_completed': []
    } for act in activations}
    
    for activation, results_list in all_results.items():
        for res in results_list:
            # Check data validity
            if (not np.isnan(res['l2re']) and not np.isinf(res['l2re']) and res['l2re'] > 0 and
                not np.isnan(res['rmse']) and not np.isinf(res['rmse']) and res['rmse'] > 0):
                
                stats[activation]['l2res'].append(res['l2re'])
                stats[activation]['rmses'].append(res['rmse'])
                stats[activation]['max_errs'].append(res['max_err'])
                stats[activation]['training_times'].append(res['training_time'])
                stats[activation]['pde_residuals'].append(res['pde_residual'])
                stats[activation]['epochs_completed'].append(res['epochs_completed'])
                
                if 'theta_mean' in res:
                    stats[activation]['theta_means'].append(res['theta_mean'])
    
    # Filter activations without valid results
    valid_activations = [act for act in activations if len(stats[act]['l2res']) > 0]
    
    if len(valid_activations) == 0:
        print("⚠️ WARNING: No valid results for any activation function!")
        return
    
    if len(valid_activations) < len(activations):
        print(f"⚠️ WARNING: Some activations failed to produce valid results:")
        failed = set(activations) - set(valid_activations)
        print(f"   Failed: {failed}")
        print(f"   Continuing with {len(valid_activations)} valid activations...\n")
    
    activations = valid_activations
    n_acts = len(activations)
    
    # Print statistics
    print(f"\n{'='*120}")
    print(f"ACTIVATION STATISTICS: {title}")
    print(f"{'='*120}")
    print(f"{'Activation':<15} {'L2RE (μ±σ)':<20} {'RMSE (μ±σ)':<20} {'Time (μ±σ)':<18} {'Epochs (μ±σ)':<18}")
    print("-" * 120)
    
    for activation in activations:
        l2res = stats[activation]['l2res']
        rmses = stats[activation]['rmses']
        times = stats[activation]['training_times']
        epochs = stats[activation]['epochs_completed']
        
        print(f"{activation.upper():<15} "
            f"{np.mean(l2res):.4e}±{np.std(l2res):.2e}  "
            f"{np.mean(rmses):.4e}±{np.std(rmses):.2e}  "
            f"{np.mean(times):.2f}±{np.std(times):.1f}s      "
            f"{int(np.mean(epochs))}±{int(np.std(epochs))}")
    
    print("=" * 120)
    
    # =================================================================
    # FIGURE 1: Main accuracy metrics (2x2)
    # =================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f'{title}: Accuracy Metrics by Activation Function', 
                fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(activations))
    
    # Plot 1.1: L2RE Comparison
    ax = axes1[0, 0]
    means = [np.mean(stats[act]['l2res']) for act in activations]
    stds = [np.std(stats[act]['l2res']) for act in activations]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, 
                color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Mark best result with red border
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('L2 Relative Error', fontsize=11, fontweight='bold')
    ax.set_title('(a) L2 Relative Error', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax.set_yscale('log')
    
    min_val = min(means)
    max_val = max([m + s for m, s in zip(means, stds)])
    ax.set_ylim(min_val * 0.3, max_val * 4)
    
    # Plot 1.2: RMSE Comparison
    ax = axes1[0, 1]
    means = [np.mean(stats[act]['rmses']) for act in activations]
    stds = [np.std(stats[act]['rmses']) for act in activations]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, 
                color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('(b) Root Mean Square Error', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax.set_yscale('log')
    
    min_val = min(means)
    max_val = max([m + s for m, s in zip(means, stds)])
    ax.set_ylim(min_val * 0.3, max_val * 4)
    
    # Plot 1.3: PDE Residual
    ax = axes1[1, 0]
    means = [np.mean(stats[act]['pde_residuals']) for act in activations]
    stds = [np.std(stats[act]['pde_residuals']) for act in activations]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, 
                color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('PDE Residual', fontsize=11, fontweight='bold')
    ax.set_title('(c) PDE Residual', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax.set_yscale('log')
    
    min_val = min(means)
    max_val = max([m + s for m, s in zip(means, stds)])
    ax.set_ylim(min_val * 0.3, max_val * 4)
    
    # Plot 1.4: Training Time
    ax = axes1[1, 1]
    means = [np.mean(stats[act]['training_times']) for act in activations]
    stds = [np.std(stats[act]['training_times']) for act in activations]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=4, 
                color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
    ax.set_title('(d) Training Time', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # FIGURE 2: Training Loss Convergence (separate)
    # =================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Classic colors (not full palette)
    classic_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f'][:n_acts]
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-'][:n_acts]
    
    for i, (activation, color, ls) in enumerate(zip(activations, classic_colors, linestyles)):
        if activation in all_results and all_results[activation] and 'history' in all_results[activation][0]:
            losses = all_results[activation][0]['history']['losses']
            
            # Filter NaN and inf
            losses = np.array(losses)
            valid_mask = np.isfinite(losses) & (losses > 0)
            if np.sum(valid_mask) > 0:
                valid_losses = losses[valid_mask]
                valid_epochs = np.where(valid_mask)[0]
                
                ax2.semilogy(valid_epochs, valid_losses, label=activation.upper(), 
                            color=color, linestyle=ls, linewidth=2, alpha=0.85)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title}: Training Loss Convergence', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--', color='gray')
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # FIGURE 3: Theta Statistics (separate)
    # =================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    
    theta_means = []
    theta_stds = []
    for activation in activations:
        if stats[activation]['theta_means']:
            theta_means.append(np.mean(stats[activation]['theta_means']))
            theta_stds.append(np.std(stats[activation]['theta_means']))
        else:
            theta_means.append(0)
            theta_stds.append(0)
    
    bars = ax3.bar(x_pos, theta_means, yerr=theta_stds, capsize=4, 
                color='steelblue', edgecolor='black', linewidth=1.2, alpha=0.8)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right', fontsize=10)
    ax3.set_ylabel('θ (mean)', fontsize=12, fontweight='bold')
    ax3.set_title(f'{title}: Average θ Parameter by Activation', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax3.set_ylim(0, 1.05)
    
    # Reference lines - classic colors
    ax3.axhline(0.0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Implicit (θ=0)')
    ax3.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Trapezoidal (θ=0.5)')
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Explicit (θ=1)')
    ax3.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # FIGURE 4: Solution Comparison
    # =================================================================
    first_results = {act: all_results[act][0] for act in activations if all_results[act]}
    
    if first_results:
        first_activation = list(first_results.keys())[0]
        
        if 'points' in first_results[first_activation] and 'u_exact' in first_results[first_activation]:
            points = first_results[first_activation]['points'].cpu()
            grid_shape = first_results[first_activation]['grid_shape']
            
            x_vals = points[:, 0].numpy().reshape(grid_shape)
            t_vals = points[:, 1].numpy().reshape(grid_shape)
            u_exact = first_results[first_activation]['u_exact'].cpu().numpy().reshape(grid_shape)
            
            times = np.linspace(t_vals[0, :].min(), t_vals[0, :].max(), 4)
            
            fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
            fig4.suptitle(f'{title}: Solution Profiles at Different Times', 
                        fontsize=14, fontweight='bold')
            
            # Classic colors for activations
            colors_sol = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f'][:n_acts]
            
            for plot_idx, t0 in enumerate(times):
                ax = axes4[plot_idx // 2, plot_idx % 2]
                idx = np.argmin(np.abs(t_vals[0, :] - t0))
                
                # Exact solution - black line
                ax.plot(x_vals[:, idx], u_exact[:, idx], 'k-', linewidth=2.5, 
                    label='Exact', alpha=1.0, zorder=10)
                
                # All activations
                for act_idx, (activation, color) in enumerate(zip(activations, colors_sol)):
                    if activation in first_results and 'u_pred' in first_results[activation]:
                        u_pred = first_results[activation]['u_pred'].cpu().numpy().reshape(grid_shape)
                        ax.plot(x_vals[:, idx], u_pred[:, idx], '--', color=color, 
                            linewidth=2, label=activation.upper(), alpha=0.8)
                
                ax.set_xlabel('x', fontsize=11, fontweight='bold')
                ax.set_ylabel('u(x,t)', fontsize=11, fontweight='bold')
                ax.set_title(f'({"abcd"[plot_idx]}) t = {t0:.3f}', fontsize=11, fontweight='bold', loc='left')
                ax.grid(True, alpha=0.3, linestyle='--', color='gray')
                
                if plot_idx == 0:
                    ax.legend(fontsize=9, loc='best', ncol=2)
            
            plt.tight_layout()
            plt.show()
    
    # Print winner
    l2re_means = [np.mean(stats[act]['l2res']) for act in activations]
    best_idx = np.argmin(l2re_means)
    best_activation = activations[best_idx]
    
    print(f"\n{'='*80}")
    print(f"BEST ACTIVATION (L2RE): {best_activation.upper()}")
    print(f"{'='*80}")
    print(f"L2RE:     {np.mean(stats[best_activation]['l2res']):.4e} ± {np.std(stats[best_activation]['l2res']):.2e}")
    print(f"RMSE:     {np.mean(stats[best_activation]['rmses']):.4e} ± {np.std(stats[best_activation]['rmses']):.2e}")
    print(f"Time:     {np.mean(stats[best_activation]['training_times']):.2f} ± {np.std(stats[best_activation]['training_times']):.1f}s")
    print(f"Residual: {np.mean(stats[best_activation]['pde_residuals']):.4e} ± {np.std(stats[best_activation]['pde_residuals']):.2e}")
    if stats[best_activation]['theta_means']:
        print(f"Theta:    {np.mean(stats[best_activation]['theta_means']):.3f} ± {np.std(stats[best_activation]['theta_means']):.3f}")
    print(f"{'='*80}\n")


# ============================================================
# MULTI-METHOD ANALYSIS AND PLOTTING
# ============================================================

def analyze_and_plot(all_results, title, domain):
    """Analysis and visualization of method comparison results"""
    methods = ['learnable_theta', 'Ordinary_PINN_15P', 'Ordinary_PINN_37P', 'Ordinary_PINN_67P']
    names = ['LF-PINN', 'Ordinary PINN 15P', 'Ordinary PINN 37P', 'Ordinary PINN 67P']
    colors = ['blue', 'green', 'orange', 'red']
    
    stats = {m: {
        'losses': [], 'l2res': [], 'rmses': [], 'max_errs': [], 
        'residuals': [], 'training_times': [], 'epochs_completed': [], 
        'converged_count': [], 'n_params': []
    } for m in methods}
    
    # Collect statistics
    for results in all_results:
        for method in methods:
            if method in results:
                res = results[method]
                stats[method]['losses'].append(res['history']['losses'][-1])
                stats[method]['l2res'].append(res['l2re'])
                stats[method]['rmses'].append(res['rmse'])
                stats[method]['max_errs'].append(res['max_err'])
                stats[method]['residuals'].append(res['pde_residual'])
                stats[method]['training_times'].append(res['training_time'])
                stats[method]['epochs_completed'].append(res['epochs_completed'])
                stats[method]['converged_count'].append(1 if res['converged'] else 0)
                stats[method]['n_params'].append(res.get('n_params', 0))
    
    # Print statistics
    print(f"\n{'='*150}")
    print(f"STATISTICS: {title}")
    print(f"{'='*150}")
    print(f"{'Method':<15} {'Params':<8} {'Loss (μ±σ)':<18} {'L2RE (μ±σ)':<18} {'RMSE (μ±σ)':<18} "
          f"{'MaxErr (μ±σ)':<18} {'Time (μ±σ)':<15} {'Epochs':<15}")
    print("-" * 150)
    
    for method, name in zip(methods, names):
        n_params = stats[method]['n_params'][0] if stats[method]['n_params'] else 0
        losses = stats[method]['losses']
        l2res = stats[method]['l2res']
        rmses = stats[method]['rmses']
        max_errs = stats[method]['max_errs']
        times = stats[method]['training_times']
        epochs_comp = stats[method]['epochs_completed']
        
        if losses:
            print(f"{name:<15} {n_params:<8} "
                  f"{np.mean(losses):.2e}±{np.std(losses):.1e}  "
                  f"{np.mean(l2res):.2e}±{np.std(l2res):.1e}  "
                  f"{np.mean(rmses):.2e}±{np.std(rmses):.1e}  "
                  f"{np.mean(max_errs):.2e}±{np.std(max_errs):.1e}  "
                  f"{np.mean(times):.1f}±{np.std(times):.1f}  "
                  f"{int(np.mean(epochs_comp))}±{int(np.std(epochs_comp))}")
    
    print("=" * 150)
    
    # =================================================================
    # FIGURE 1: Main metrics (2x3)
    # =================================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(f'{title}: Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(names))
    
    # Plot 1: L2RE Comparison
    ax = axes1[0, 0]
    means = [np.mean(stats[m]['l2res']) for m in methods]
    stds = [np.std(stats[m]['l2res']) for m in methods]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('L2 Relative Error', fontsize=12)
    ax.set_title('L2RE Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Plot 2: RMSE Comparison
    ax = axes1[0, 1]
    means = [np.mean(stats[m]['rmses']) for m in methods]
    stds = [np.std(stats[m]['rmses']) for m in methods]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('RMSE', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Plot 3: PDE Residual
    ax = axes1[0, 2]
    means = [np.mean(stats[m]['residuals']) for m in methods]
    stds = [np.std(stats[m]['residuals']) for m in methods]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('PDE Residual', fontsize=12)
    ax.set_title('PDE Residual', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Plot 4: Training Time
    ax = axes1[1, 0]
    means = [np.mean(stats[m]['training_times']) for m in methods]
    stds = [np.std(stats[m]['training_times']) for m in methods]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Training Time (s)', fontsize=12)
    ax.set_title('Training Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Theta Evolution
    ax = axes1[1, 1]
    if 'learnable_theta' in all_results[0] and 'history' in all_results[0]['learnable_theta']:
        history = all_results[0]['learnable_theta']['history']
        if 'theta_statistics' in history and history['theta_statistics']:
            theta_stats = [s for s in history['theta_statistics'] if s is not None]
            
            if theta_stats:
                theta_means = [s['mean'] for s in theta_stats]
                theta_stds = [s['std'] for s in theta_stats]
                epochs = range(len(theta_means))
                
                ax.plot(epochs, theta_means, 'r-', linewidth=3, label='θ (mean)', alpha=0.8)
                ax.fill_between(epochs, np.array(theta_means) - np.array(theta_stds),
                               np.array(theta_means) + np.array(theta_stds),
                               alpha=0.2, color='red', label='±σ')
                
                ax.axhline(0.0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Implicit')
                ax.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Trapezoidal')
                ax.axhline(1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Explicit')
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('θ', fontsize=12)
                ax.set_title('θ Evolution (LF-PINN)', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.1, 1.1)
    
    # Plot 6: Max Error
    ax = axes1[1, 2]
    means = [np.mean(stats[m]['max_errs']) for m in methods]
    stds = [np.std(stats[m]['max_errs']) for m in methods]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title('Maximum Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # FIGURE 2: Training Loss Convergence (separate)
    # =================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    for method, name, color in zip(methods, names, colors):
        if method in all_results[0] and 'history' in all_results[0][method]:
            losses = all_results[0][method]['history']['losses']
            ax2.semilogy(losses, label=name, color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title}: Training Loss Convergence', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =================================================================
    # FIGURE 3: Solution comparison
    # =================================================================
    plot_solutions(all_results[0], title, domain)


# ============================================================
# SOLUTION VISUALIZATION
# ============================================================

def plot_solutions(results, title, domain):
    """Visualization of solutions in time slices"""
    methods = ['learnable_theta', 'Ordinary_PINN_15P', 'Ordinary_PINN_37P', 'Ordinary_PINN_67P']
    names = ['LF-PINN', 'Ordinary PINN 15P', 'Ordinary PINN 37P', 'Ordinary PINN 67P']
    colors = ['blue', 'green', 'orange', 'red']
    linestyles = ['-', '--', '-.', ':']
    
    if 'learnable_theta' not in results or 'u_exact' not in results['learnable_theta']:
        return
    
    points = results['learnable_theta']['points'].cpu()
    grid_shape = results['learnable_theta']['grid_shape']
    
    x_vals = points[:, 0].numpy().reshape(grid_shape)
    t_vals = points[:, 1].numpy().reshape(grid_shape)
    u_exact = results['learnable_theta']['u_exact'].cpu().numpy().reshape(grid_shape)
    
    times = np.linspace(t_vals[0, :].min(), t_vals[0, :].max(), 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title}: Solutions u(x,t)', fontsize=16, fontweight='bold')
    
    for i, t0 in enumerate(times):
        ax = axes[i]
        idx = np.argmin(np.abs(t_vals[0, :] - t0))
        
        # Exact solution
        ax.plot(x_vals[:, idx], u_exact[:, idx], 'k-', linewidth=3.5, 
                label='Exact', alpha=0.9, zorder=10)
        
        # All methods
        for method, name, color, ls in zip(methods, names, colors, linestyles):
            if method in results and 'u_pred' in results[method]:
                u_pred = results[method]['u_pred'].cpu().numpy().reshape(grid_shape)
                ax.plot(x_vals[:, idx], u_pred[:, idx], linestyle=ls, color=color, 
                       linewidth=2, label=name, alpha=0.75)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f't = {t0:.3f}', fontsize=13)
        if i == 0:
            ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# FINAL COMPARISON (ALL PDEs)
# ============================================================

def plot_final_comparison(all_pde_results):
    """Final comparison across all PDEs"""
    methods = ['learnable_theta', 'Ordinary_PINN_15P', 'Ordinary_PINN_37P', 'Ordinary_PINN_67P']
    names = ['Modified_LF_PINN', 'Ordinary_PINN_15P', 'Ordinary_PINN_37P', 'Ordinary_PINN_67P']
    colors = ['blue', 'green', 'orange', 'red']
    pde_names = ['Heat', 'Wave', 'Burgers', 'Reaction-Diff']
    
    # === Plot 1: L2RE comparison for all PDEs ===
    fig, ax = plt.subplots(figsize=(16, 8))
    x_pos = np.arange(len(pde_names))
    bar_width = 0.2  # Reduced width for 4 methods
    
    for i, (method, name, color) in enumerate(zip(methods, names, colors)):
        means, stds = [], []
        for pde_type, results_list in all_pde_results.items():
            l2res = [r[method]['l2re'] for r in results_list if method in r]
            means.append(np.mean(l2res) if l2res else 0)
            stds.append(np.std(l2res) if l2res else 0)
        
        ax.bar(x_pos + i*bar_width, means, bar_width, yerr=stds, label=name, 
               color=color, alpha=0.7, capsize=3, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('PDE Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('L2 Relative Error (μ±σ)', fontsize=14, fontweight='bold')
    ax.set_title('All Methods on All PDEs: L2RE', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos + bar_width*1.5)  # Center labels
    ax.set_xticklabels(pde_names, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # === Plot 2: Training Time comparison ===
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for i, (method, name, color) in enumerate(zip(methods, names, colors)):
        means, stds = [], []
        for pde_type, results_list in all_pde_results.items():
            times = [r[method]['training_time'] for r in results_list if method in r]
            means.append(np.mean(times) if times else 0)
            stds.append(np.std(times) if times else 0)
        
        ax.bar(x_pos + i*bar_width, means, bar_width, yerr=stds, label=name, 
               color=color, alpha=0.7, capsize=3, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('PDE Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Time (s) (μ±σ)', fontsize=14, fontweight='bold')
    ax.set_title('All Methods on All PDEs: Training Time', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos + bar_width*1.5)
    ax.set_xticklabels(pde_names, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # === Plot 3: Heatmap L2RE ===
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data for heatmap
    heatmap_data = np.zeros((len(pde_names), len(methods)))
    
    for i, pde_type in enumerate(all_pde_results.keys()):
        results_list = all_pde_results[pde_type]
        for j, method in enumerate(methods):
            l2res = [r[method]['l2re'] for r in results_list if method in r]
            heatmap_data[i, j] = np.mean(l2res) if l2res else 0
    
    # Log scale for better visibility
    heatmap_data_log = np.log10(heatmap_data + 1e-10)
    
    im = ax.imshow(heatmap_data_log, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(pde_names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=11)
    ax.set_yticklabels(pde_names, fontsize=11)
    
    # Add values to cells
    for i in range(len(pde_names)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2e}',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold')
    
    ax.set_title('L2RE Heatmap: PDE × Method (log scale)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(L2RE)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # === Plot 4: Accuracy vs Parameters for all PDEs ===
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pde_markers = ['o', 's', '^', 'D']  # Different markers for PDEs
    
    for pde_idx, (pde_type, results_list) in enumerate(all_pde_results.items()):
        pde_name = pde_names[pde_idx]
        marker = pde_markers[pde_idx]
        
        for method_idx, method in enumerate(methods):
            if method in results_list[0]:
                n_params = results_list[0][method]['n_params']
                l2res = [r[method]['l2re'] for r in results_list if method in r]
                l2re_mean = np.mean(l2res)
                l2re_std = np.std(l2res)
                
                color = colors[method_idx]
                
                ax.errorbar(n_params, l2re_mean, yerr=l2re_std,
                           fmt=marker, markersize=10, capsize=4,
                           color=color, alpha=0.7, linewidth=2,
                           label=f'{pde_name}-{names[method_idx]}' if pde_idx == 0 and method_idx == 0 else '')
        
    # Legend for methods (colors)
    method_handles = [Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=c, markersize=10, label=n)
                     for c, n in zip(colors, names)]
    
    # Legend for PDEs (markers)
    pde_handles = [Line2D([0], [0], marker=m, color='gray', 
                         markersize=10, label=p, linestyle='None')
                  for m, p in zip(pde_markers, pde_names)]
    
    first_legend = ax.legend(handles=method_handles, title='Methods', 
                            loc='upper right', fontsize=10)
    ax.add_artist(first_legend)
    ax.legend(handles=pde_handles, title='PDEs', 
             loc='lower left', fontsize=10)
    
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('L2 Relative Error', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs Model Size (All PDEs)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# FORWARD PROBLEM COMPARISON
# ============================================================

def plot_forward_comparison(histories, results, pde_title):
    """
    Visualization for data-assisted forward problem.
    
    histories: {'name': history, ...}
    results: {'name': {'l2re': ..., 'rmse': ...}, ...}
    """
    names = list(histories.keys())
    n_models = len(names)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Data-Assisted Forward Problem: {pde_title}', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # 1. Loss vs Time
    ax1 = axes[0]
    for i, name in enumerate(names):
        ax1.semilogy(histories[name]['time'], histories[name]['total_loss'], 
                     color=colors[i], linewidth=2, label=name)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. L2RE comparison (bar chart)
    ax2 = axes[1]
    l2re_vals = [results[name]['l2re'] for name in names]
    bars = ax2.bar(names, l2re_vals, color=colors[:n_models])
    ax2.set_ylabel('L2 Relative Error')
    ax2.set_title('Final Accuracy (L2RE)')
    ax2.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, l2re_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 3. RMSE comparison
    ax3 = axes[2]
    rmse_vals = [results[name]['rmse'] for name in names]
    bars = ax3.bar(names, rmse_vals, color=colors[:n_models])
    ax3.set_ylabel('RMSE')
    ax3.set_title('Final Accuracy (RMSE)')
    ax3.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, rmse_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.2e}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Table
    print(f"\n{'='*70}")
    print(f"{'Model':<30} {'L2RE':<15} {'RMSE':<15} {'Epochs':<10}")
    print(f"{'-'*70}")
    for name in names:
        l2re = results[name]['l2re']
        rmse = results[name]['rmse']
        epochs = histories[name]['total_epochs']
        print(f"{name:<30} {l2re:<15.4e} {rmse:<15.4e} {epochs:<10}")
    print(f"{'='*70}")


# ============================================================
# LEARNING RATE VISUALIZATION
# ============================================================

def plot_lr_comparison(results: dict, pde_title: str, metric: str = 'l2re'):
    """
    Visualization of learning rate comparison (without data).
    
    Args:
        results: dict from test_lr_single_pde
        pde_title: PDE name for title
        metric: 'l2re' or 'rmse'
    """
    lr_values = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{pde_title}: Learning Rate Study', fontsize=14, fontweight='bold')
    
    # 1. Bar chart with error bars
    ax1 = axes[0]
    means = [results[lr][f'{metric}_mean'] for lr in lr_values]
    stds = [results[lr][f'{metric}_std'] for lr in lr_values]
    
    x_pos = np.arange(len(lr_values))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{lr}' for lr in lr_values], rotation=45)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'{metric.upper()} by Learning Rate')
    ax1.set_yscale('log')
    
    # Value annotations
    for bar, mean, std in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{mean:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 2. Loss curves (example from first run)
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lr_values)))
    
    for i, lr in enumerate(lr_values):
        if results[lr]['histories']:
            history = results[lr]['histories'][0]  # first run
            if history['losses']:
                ax2.semilogy(history['losses'], color=colors[i], 
                            label=f'lr={lr}', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Run 1)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Epochs completed
    ax3 = axes[2]
    epochs = [results[lr]['epochs_mean'] for lr in lr_values]
    ax3.bar(x_pos, epochs, color='coral', alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{lr}' for lr in lr_values], rotation=45)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Epochs')
    ax3.set_title('Epochs Completed (avg)')
    
    for i, ep in enumerate(epochs):
        ax3.text(i, ep, f'{int(ep)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Table
    print(f"\n{'='*80}")
    print(f"{'LR':<12} {metric.upper()+' Mean':<15} {metric.upper()+' Std':<15} {'Epochs':<12} {'Final Loss':<15}")
    print(f"{'-'*80}")
    for lr in lr_values:
        r = results[lr]
        print(f"{lr:<12} {r[f'{metric}_mean']:<15.4e} {r[f'{metric}_std']:<15.4e} "
              f"{r['epochs_mean']:<12.0f} {r['final_loss_mean']:<15.4e}")
    print(f"{'='*80}")
    
    # Best LR
    best_lr = min(lr_values, key=lambda x: results[x][f'{metric}_mean'])
    print(f"\n✓ Best LR: {best_lr} with {metric.upper()} = {results[best_lr][f'{metric}_mean']:.4e}")


# ============================================================
# DATA FINETUNING LR VISUALIZATION
# ============================================================

def plot_lr_finetuning_comparison(results: dict, pde_title: str):
    """
    Visualization of LR study results for data finetuning.
    
    Args:
        results: dict from test_lr_data_finetuning
        pde_title: PDE name
    """
    # Separate baseline from lr results
    baseline = results.get('baseline', None)
    lr_values = sorted([k for k in results.keys() if isinstance(k, float)])
    
    if not lr_values:
        print("No LR results to plot!")
        return
    
    l2re_means = [results[lr]['l2re_mean'] for lr in lr_values]
    l2re_stds = [results[lr]['l2re_std'] for lr in lr_values]
    improvement_means = [results[lr]['improvement_mean'] for lr in lr_values]
    improvement_stds = [results[lr]['improvement_std'] for lr in lr_values]
    epochs_means = [results[lr]['finetune_epochs_mean'] for lr in lr_values]
    data_loss_means = [results[lr]['final_data_loss_mean'] for lr in lr_values]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'LR Finetuning Study: {pde_title} (LF-PINN)', fontsize=14, fontweight='bold')
    
    lr_labels = [f'{lr:.0e}' if lr < 0.001 else f'{lr}' for lr in lr_values]
    x_pos = np.arange(len(lr_values))
    
    # 1. L2RE vs LR (with baseline)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=5, color='steelblue', alpha=0.8)
    if baseline:
        ax1.axhline(baseline['l2re_mean'], color='red', linestyle='--', linewidth=2, label='Before finetuning')
        ax1.fill_between([-0.5, len(lr_values)-0.5], 
                         baseline['l2re_mean'] - baseline['l2re_std'],
                         baseline['l2re_mean'] + baseline['l2re_std'],
                         color='red', alpha=0.1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(lr_labels, rotation=45)
    ax1.set_xlabel('Learning Rate (finetuning)')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('Final Accuracy (L2RE)')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    best_idx = np.nanargmin(l2re_means)
    bars1[best_idx].set_color('green')
    
    # 2. Improvement vs LR
    ax2 = axes[0, 1]
    colors2 = ['green' if imp > 0 else 'red' for imp in improvement_means]
    bars2 = ax2.bar(x_pos, improvement_means, yerr=improvement_stds, capsize=5, color=colors2, alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(lr_labels, rotation=45)
    ax2.set_xlabel('Learning Rate (finetuning)')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Accuracy Improvement from Finetuning')
    ax2.grid(True, alpha=0.3)
    
    # Annotations for improvement
    for i, (bar, imp) in enumerate(zip(bars2, improvement_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{imp:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Finetune loss curves (first run for each lr)
    ax3 = axes[1, 0]
    colors_curve = plt.cm.viridis(np.linspace(0, 1, len(lr_values)))
    
    for i, lr in enumerate(lr_values):
        history = results[lr]['histories'][0]
        if history['data_loss']:
            ax3.semilogy(history['data_loss'], color=colors_curve[i],
                        label=f'lr={lr_labels[i]}', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Finetuning Epoch')
    ax3.set_ylabel('Data Loss (log scale)')
    ax3.set_title('Finetuning Loss Curves (Run 1)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Final data loss vs LR
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, data_loss_means, color='coral', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(lr_labels, rotation=45)
    ax4.set_xlabel('Learning Rate (finetuning)')
    ax4.set_ylabel('Final Data Loss')
    ax4.set_title('Data Fitting Quality')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Mark best
    bars4[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.show()
    
    # Table
    print(f"\n{'='*100}")
    print(f"LR Finetuning Study: {pde_title} (LF-PINN)")
    print(f"{'='*100}")
    print(f"{'LR':<12} {'L2RE After':<24} {'Improvement':<20} {'Epochs':<12} {'Data Loss':<15}")
    print(f"{'-'*100}")
    
    if baseline:
        print(f"{'baseline':<12} {baseline['l2re_mean']:.2e} ± {baseline['l2re_std']:.2e}         "
              f"{'--':<20} {'--':<12} {'--':<15}")
    
    best_lr = lr_values[best_idx]
    for lr in lr_values:
        r = results[lr]
        l2re_str = f"{r['l2re_mean']:.2e} ± {r['l2re_std']:.2e}"
        imp_str = f"{r['improvement_mean']:+.1f}% ± {r['improvement_std']:.1f}%"
        marker = " ← BEST" if lr == best_lr else ""
        print(f"{lr:<12.5f} {l2re_str:<24} {imp_str:<20} {r['finetune_epochs_mean']:<12.0f} "
              f"{r['final_data_loss_mean']:<15.2e}{marker}")
    
    print(f"{'='*100}")
    
    return best_lr


# ============================================================
# DATA TRAINING EPOCHS SATURATION STUDY
# ============================================================

def plot_data_tratining_epochs_study(results: dict, pde_title: str):
    """Visualization of finetuning epochs study."""
    
    epochs = results['epochs']
    l2re_mean = results['l2re_mean']
    l2re_std = results['l2re_std']
    data_loss_mean = results['data_loss_mean']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Finetuning Epochs Study: {pde_title}', fontsize=14, fontweight='bold')
    
    # 1. L2RE vs Epochs
    ax1 = axes[0]
    ax1.plot(epochs, l2re_mean, 'b-', linewidth=2, label='L2RE (mean)')
    ax1.fill_between(epochs, 
                     np.array(l2re_mean) - np.array(l2re_std),
                     np.array(l2re_mean) + np.array(l2re_std),
                     alpha=0.3, color='blue')
    
    # Find saturation point (where improvement < 1% of initial)
    baseline = l2re_mean[0]
    for i, (e, l) in enumerate(zip(epochs, l2re_mean)):
        if i > 0:
            improvement = (l2re_mean[i-1] - l) / baseline * 100
            if improvement < 0.5:  # less than 0.5% improvement
                ax1.axvline(e, color='red', linestyle='--', alpha=0.7, label=f'Saturation ~{e} epochs')
                break
    
    ax1.axhline(baseline, color='gray', linestyle=':', alpha=0.5, label='Before finetuning')
    ax1.set_xlabel('Finetuning Epochs')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('Accuracy vs Finetuning Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Data Loss vs Epochs
    ax2 = axes[1]
    ax2.semilogy(epochs[1:], data_loss_mean[1:], 'r-', linewidth=2)
    ax2.set_xlabel('Finetuning Epochs')
    ax2.set_ylabel('Data Loss (log)')
    ax2.set_title('Data Loss vs Finetuning Epochs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"Finetuning Epochs Study: {pde_title}")
    print(f"{'='*60}")
    print(f"{'Epochs':<12} {'L2RE':<20} {'Improvement':<15}")
    print(f"{'-'*60}")
    
    for i, e in enumerate(epochs):
        if i % 5 == 0 or i == len(epochs) - 1:  # every 5 points
            imp = (baseline - l2re_mean[i]) / baseline * 100
            print(f"{e:<12} {l2re_mean[i]:.4e} ± {l2re_std[i]:.4e}   {imp:+.1f}%")
    
    print(f"{'='*60}")


# ============================================================
# INITIAL THETA COMPARISON
# ============================================================

def plot_initial_theta_comparison(all_results: dict, title: str):
    """
    Visualization of initial theta value comparison.
    """
    theta_values = sorted(all_results.keys())
    
    # Collect statistics
    l2re_means, l2re_stds = [], []
    theta_final_means, theta_final_stds = [], []
    
    for theta in theta_values:
        l2res = [r['l2re'] for r in all_results[theta]]
        l2re_means.append(np.mean(l2res))
        l2re_stds.append(np.std(l2res))
        
        theta_finals = [r['theta_final_mean'] for r in all_results[theta]]
        theta_final_means.append(np.mean(theta_finals))
        theta_final_stds.append(np.std(theta_finals))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Initial Theta Study: {title}', fontsize=14, fontweight='bold')
    
    theta_labels = [f'{t:.2f}' for t in theta_values]
    x_pos = np.arange(len(theta_values))
    
    # 1. L2RE comparison
    ax1 = axes[0]
    bars = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=5, color='steelblue', alpha=0.8)
    best_idx = np.argmin(l2re_means)
    bars[best_idx].set_color('green')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(theta_labels)
    ax1.set_xlabel('Initial θ')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('Final Accuracy (L2RE)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Annotations
    for i, (bar, mean, std) in enumerate(zip(bars, l2re_means, l2re_stds)):
        marker = " ← BEST" if i == best_idx else ""
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{mean:.2e}{marker}', ha='center', va='bottom', fontsize=9)
    
    # 2. Final theta comparison
    ax2 = axes[1]
    ax2.bar(x_pos, theta_final_means, yerr=theta_final_stds, capsize=5, color='coral', alpha=0.8)
    ax2.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Trapezoidal')
    ax2.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit')
    ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(theta_labels)
    ax2.set_xlabel('Initial θ')
    ax2.set_ylabel('Final θ (mean)')
    ax2.set_title('Learned θ After Training')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training curves (first run)
    ax3 = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_values)))
    
    for i, theta in enumerate(theta_values):
        history = all_results[theta][0]['history']
        if 'losses' in history and history['losses']:
            ax3.semilogy(history['losses'], color=colors[i], 
                        label=f'θ₀={theta:.2f}', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log)')
    ax3.set_title('Training Loss (Run 1)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Results table
    print(f"\n{'='*90}")
    print(f"SUMMARY: {title} - Initial Theta Study")
    print(f"{'='*90}")
    print(f"{'Initial θ':<15} {'L2RE':<24} {'Final θ':<20} {'Converged to':<15}")
    print(f"{'-'*90}")
    
    best_theta = theta_values[best_idx]
    for i, theta in enumerate(theta_values):
        l2re_str = f"{l2re_means[i]:.2e} ± {l2re_stds[i]:.2e}"
        theta_str = f"{theta_final_means[i]:.3f} ± {theta_final_stds[i]:.3f}"
        
        # Determine convergence target
        final = theta_final_means[i]
        if final < 0.15:
            converged = "~Implicit"
        elif final > 0.85:
            converged = "~Explicit"
        elif 0.4 < final < 0.6:
            converged = "~Trapezoidal"
        else:
            converged = f"Custom ({final:.2f})"
        
        marker = " ← BEST" if theta == best_theta else ""
        print(f"{theta:<15.2f} {l2re_str:<24} {theta_str:<20} {converged:<15}{marker}")
    
    print(f"{'='*90}")
    
    return best_theta


# ============================================================
# OPTIMIZER COMPARISON
# ============================================================

def plot_optimizers_comparison(all_results: dict, title: str):
    """
    Visualization of optimizer comparison results.
    
    Args:
        all_results: dict with results for each optimizer
        title: plot title (PDE name)
    
    Returns:
        best_optimizer: name of best optimizer by L2RE
    """
    optimizers = list(all_results.keys())
    
    # Collect statistics
    l2re_means, l2re_stds = [], []
    time_means, time_stds = [], []
    theta_final_means = []
    
    for opt in optimizers:
        l2res = [r['l2re'] for r in all_results[opt]]
        l2re_means.append(np.mean(l2res))
        l2re_stds.append(np.std(l2res))
        
        times = [r['training_time'] for r in all_results[opt]]
        time_means.append(np.mean(times))
        time_stds.append(np.std(times))
        
        theta_finals = [r.get('theta_final_mean', 0.5) for r in all_results[opt]]
        theta_final_means.append(np.mean(theta_finals))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Optimizer Comparison: {title}', fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(optimizers))
    
    # 1. L2RE comparison (bar chart)
    ax1 = axes[0, 0]
    bars = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=4, color='steelblue', alpha=0.8)
    best_idx = np.argmin(l2re_means)
    bars[best_idx].set_color('green')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(optimizers, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('Final Accuracy (L2RE)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotations
    for i, (bar, mean) in enumerate(zip(bars, l2re_means)):
        marker = " ★" if i == best_idx else ""
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                f'{mean:.2e}{marker}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # 2. Training time comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, time_means, yerr=time_stds, capsize=4, color='coral', alpha=0.8)
    fastest_idx = np.argmin(time_means)
    bars2[fastest_idx].set_color('green')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(optimizers, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Training Time (s)')
    ax2.set_title('Training Time')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Training curves (first run, all optimizers)
    ax3 = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
    
    for i, opt in enumerate(optimizers):
        history = all_results[opt][0]['history']
        if 'losses' in history and history['losses']:
            ax3.semilogy(history['losses'], color=colors[i], 
                        label=opt, linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log)')
    ax3.set_title('Training Loss Curves (Run 1)')
    ax3.legend(fontsize=8, ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Final theta comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, theta_final_means, color='mediumpurple', alpha=0.8)
    ax4.axhline(0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Trapezoidal (0.5)')
    ax4.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit (0)')
    ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit (1)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(optimizers, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Final θ (mean)')
    ax4.set_title('Learned θ After Training')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY: {title} - Optimizer Comparison")
    print(f"{'='*100}")
    print(f"{'Optimizer':<18} {'L2RE':<24} {'Time (s)':<16} {'Final θ':<12} {'Note':<15}")
    print(f"{'-'*100}")
    
    best_opt = optimizers[best_idx]
    for i, opt in enumerate(optimizers):
        l2re_str = f"{l2re_means[i]:.2e} ± {l2re_stds[i]:.2e}"
        time_str = f"{time_means[i]:.1f} ± {time_stds[i]:.1f}"
        theta_str = f"{theta_final_means[i]:.3f}"
        
        # Notes
        notes = []
        if i == best_idx:
            notes.append("BEST L2RE")
        if i == fastest_idx:
            notes.append("FASTEST")
        note_str = ", ".join(notes) if notes else ""
        
        print(f"{opt:<18} {l2re_str:<24} {time_str:<16} {theta_str:<12} {note_str:<15}")
    
    print(f"{'='*100}")
    
    # Efficiency analysis
    print(f"\n{'='*60}")
    print("EFFICIENCY ANALYSIS (L2RE per second)")
    print(f"{'='*60}")
    
    efficiency = [l2re / t for l2re, t in zip(l2re_means, time_means)]
    sorted_idx = np.argsort(efficiency)
    
    for rank, idx in enumerate(sorted_idx):
        eff_str = f"{efficiency[idx]:.2e}"
        print(f"  {rank+1}. {optimizers[idx]:<15} efficiency={eff_str} (L2RE={l2re_means[idx]:.2e}, time={time_means[idx]:.1f}s)")
    
    print(f"{'='*60}")
    
    return best_opt

def plot_optimizer_data_finetuning(results: Dict, pde_type: str, lr: float,
                                    save_path: str = None):
    """
    Plot optimizer comparison for data finetuning.
    
    4 subplots:
    1. L2RE comparison (bar chart)
    2. Improvement % (bar chart)
    3. Data loss curves (line plot)
    4. Final data loss comparison (bar chart)
    """
    # Get optimizer names (exclude baseline)
    opt_names = [k for k in results.keys() if k != 'baseline']
    n_opts = len(opt_names)
    x_pos = np.arange(n_opts)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Optimizer Comparison for Data Finetuning (lr={lr})', 
                 fontsize=14, fontweight='bold')
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_opts))
    
    # 1. L2RE comparison
    ax1 = axes[0, 0]
    l2re_means = [results[opt]['l2re_mean'] for opt in opt_names]
    l2re_stds = [results[opt]['l2re_std'] for opt in opt_names]
    
    bars1 = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=5, 
                    color=colors, alpha=0.8, edgecolor='black')
    
    # Baseline line
    ax1.axhline(results['baseline']['l2re_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Baseline: {results['baseline']['l2re_mean']:.2e}")
    
    # Highlight best
    best_idx = np.argmin(l2re_means)
    bars1[best_idx].set_edgecolor('green')
    bars1[best_idx].set_linewidth(3)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(opt_names, rotation=45, ha='right')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE After Finetuning')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, mean, std) in enumerate(zip(bars1, l2re_means, l2re_stds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{mean:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 2. Improvement %
    ax2 = axes[0, 1]
    imp_means = [results[opt]['improvement_mean'] for opt in opt_names]
    imp_stds = [results[opt]['improvement_std'] for opt in opt_names]
    
    bars2 = ax2.bar(x_pos, imp_means, yerr=imp_stds, capsize=5,
                    color=colors, alpha=0.8, edgecolor='black')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # Color bars by positive/negative improvement
    for i, (bar, imp) in enumerate(zip(bars2, imp_means)):
        if imp < 0:
            bar.set_color('red')
            bar.set_alpha(0.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(opt_names, rotation=45, ha='right')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Improvement vs Baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values
    for bar, imp in zip(bars2, imp_means):
        y_pos = bar.get_height() + 0.5 if imp >= 0 else bar.get_height() - 2
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    # 3. Data loss curves (example run)
    ax3 = axes[1, 0]
    
    for i, opt in enumerate(opt_names):
        if results[opt]['histories'] and results[opt]['histories'][0]['data_loss']:
            losses = results[opt]['histories'][0]['data_loss']
            ax3.semilogy(losses, label=opt, color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Data Loss')
    ax3.set_title('Data Loss During Finetuning (Example Run)')
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Final data loss
    ax4 = axes[1, 1]
    final_losses = [results[opt]['final_data_loss_mean'] for opt in opt_names]
    
    bars4 = ax4.bar(x_pos, final_losses, color=colors, alpha=0.8, edgecolor='black')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(opt_names, rotation=45, ha='right')
    ax4.set_ylabel('Final Data Loss')
    ax4.set_title('Final Data Loss')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig
# ============================================================
# OPTIMIZER DETAILED COMPARISON
# ============================================================

def plot_optimizers_detailed(all_results: dict, title: str, top_n: int = 4):
    """
    Detailed comparison of top N optimizers with loss component breakdown.
    
    Args:
        all_results: dict with results for each optimizer
        title: plot title
        top_n: number of top optimizers to show in detail
    """
    optimizers = list(all_results.keys())
    
    # Sort by L2RE
    l2re_means = [np.mean([r['l2re'] for r in all_results[opt]]) for opt in optimizers]
    sorted_indices = np.argsort(l2re_means)
    top_optimizers = [optimizers[i] for i in sorted_indices[:top_n]]
    
    fig, axes = plt.subplots(2, top_n, figsize=(4*top_n, 8))
    fig.suptitle(f'Top {top_n} Optimizers Detail: {title}', fontsize=14, fontweight='bold')
    
    colors = plt.cm.Set2(np.linspace(0, 1, 5))
    
    for col, opt in enumerate(top_optimizers):
        history = all_results[opt][0]['history']
        
        # Top row: Loss curves
        ax_top = axes[0, col]
        if 'losses' in history:
            ax_top.semilogy(history['losses'], color='blue', label='Total', linewidth=2)
        if 'pde_losses' in history:
            ax_top.semilogy(history['pde_losses'], color='red', label='PDE', linewidth=1.5, alpha=0.7)
        
        l2re_mean = np.mean([r['l2re'] for r in all_results[opt]])
        ax_top.set_title(f'{opt}\nL2RE={l2re_mean:.2e}', fontsize=10)
        ax_top.set_xlabel('Epoch')
        if col == 0:
            ax_top.set_ylabel('Loss (log)')
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)
        
        # Bottom row: L2RE distribution across runs
        ax_bot = axes[1, col]
        l2res = [r['l2re'] for r in all_results[opt]]
        ax_bot.boxplot(l2res, vert=True)
        ax_bot.scatter([1]*len(l2res), l2res, alpha=0.5, color='steelblue')
        ax_bot.set_ylabel('L2RE' if col == 0 else '')
        ax_bot.set_title(f'Distribution (n={len(l2res)})', fontsize=10)
        ax_bot.set_xticks([])
        ax_bot.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_theta_hidden_dim_comparison(results: Dict, pde_type: str, 
                                      save_path: str = None):
    """
    Plot comparison of different theta_hidden_dim values.
    
    6 subplots:
    1. L2RE vs hidden_dim
    2. L2RE vs n_params
    3. Training time vs hidden_dim
    4. Final theta vs hidden_dim
    5. Loss curves (example run)
    6. L2RE box plot
    """
    hidden_dims = sorted(results.keys())
    
    def count_theta_params(h):
        return 6 * h + 1
    
    # Collect statistics
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
            'l2re_all': l2res,
            'rmse_mean': np.mean(rmses),
            'theta_mean': np.mean(thetas),
            'theta_std': np.std(thetas),
            'time_mean': np.mean(times),
            'time_std': np.std(times)
        })
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Theta Hidden Dim Comparison', 
                 fontsize=14, fontweight='bold')
    
    x_dims = [s['hidden_dim'] for s in stats]
    x_params = [s['n_params'] for s in stats]
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(hidden_dims)))
    best_idx = np.argmin([s['l2re_mean'] for s in stats])
    
    # 1. L2RE vs hidden_dim
    ax1 = axes[0, 0]
    l2re_means = [s['l2re_mean'] for s in stats]
    l2re_stds = [s['l2re_std'] for s in stats]
    
    ax1.errorbar(x_dims, l2re_means, yerr=l2re_stds, fmt='o-', 
                 color='steelblue', linewidth=2, markersize=8, capsize=5)
    ax1.scatter([x_dims[best_idx]], [l2re_means[best_idx]], 
                color='green', s=200, zorder=5, marker='*', label='Best')
    
    ax1.set_xlabel('theta_hidden_dim')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE vs Hidden Dim')
    ax1.set_yscale('log')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. L2RE vs n_params
    ax2 = axes[0, 1]
    ax2.errorbar(x_params, l2re_means, yerr=l2re_stds, fmt='s-', 
                 color='coral', linewidth=2, markersize=8, capsize=5)
    ax2.scatter([x_params[best_idx]], [l2re_means[best_idx]], 
                color='green', s=200, zorder=5, marker='*', label='Best')
    
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('L2 Relative Error')
    ax2.set_title('L2RE vs Parameter Count')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add parameter labels
    for i, (x, y, h) in enumerate(zip(x_params, l2re_means, x_dims)):
        ax2.annotate(f'h={h}', (x, y), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=8)
    
    # 3. Training time vs hidden_dim
    ax3 = axes[0, 2]
    time_means = [s['time_mean'] for s in stats]
    time_stds = [s['time_std'] for s in stats]
    
    ax3.bar(range(len(hidden_dims)), time_means, yerr=time_stds, 
            color=colors, alpha=0.8, edgecolor='black', capsize=5)
    ax3.set_xticks(range(len(hidden_dims)))
    ax3.set_xticklabels([f'h={h}\n({count_theta_params(h)}p)' for h in hidden_dims], fontsize=9)
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Time vs Hidden Dim')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Final theta vs hidden_dim
    ax4 = axes[1, 0]
    theta_means = [s['theta_mean'] for s in stats]
    theta_stds = [s['theta_std'] for s in stats]
    
    ax4.errorbar(x_dims, theta_means, yerr=theta_stds, fmt='D-', 
                 color='purple', linewidth=2, markersize=8, capsize=5)
    
    # Reference lines
    ax4.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Trapezoidal (0.5)')
    ax4.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit (0.0)')
    ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit (1.0)')
    
    ax4.set_xlabel('theta_hidden_dim')
    ax4.set_ylabel('Final θ (mean)')
    ax4.set_title('Learned θ vs Hidden Dim')
    ax4.set_xscale('log', base=2)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss curves (example run)
    ax5 = axes[1, 1]
    
    for i, h in enumerate(hidden_dims):
        if results[h] and results[h][0].get('history') and results[h][0]['history'].get('losses'):
            losses = results[h][0]['history']['losses']
            ax5.semilogy(losses, label=f'h={h}', color=colors[i], linewidth=1.5, alpha=0.8)
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Total Loss')
    ax5.set_title('Training Loss (Example Run)')
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # 6. L2RE box plot
    ax6 = axes[1, 2]
    
    l2re_data = [s['l2re_all'] for s in stats]
    bp = ax6.boxplot(l2re_data, labels=[f'h={h}' for h in hidden_dims], 
                     patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight best
    bp['boxes'][best_idx].set_edgecolor('green')
    bp['boxes'][best_idx].set_linewidth(3)
    
    ax6.set_ylabel('L2 Relative Error')
    ax6.set_title('L2RE Distribution')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_input_features_comparison(results: Dict, feature_sets: Dict, 
                                    pde_type: str, save_path: str = None):
    """
    Plot comparison of different input feature sets.
    
    4 subplots:
    1. L2RE comparison (bar chart, sorted)
    2. L2RE vs number of inputs
    3. Training time comparison
    4. Loss curves (example run)
    """
    set_names = list(feature_sets.keys())
    
    # Collect statistics
    stats = []
    for name in set_names:
        runs = results[name]
        l2res = [r['l2re'] for r in runs]
        times = [r['training_time'] for r in runs]
        
        stats.append({
            'name': name,
            'n_inputs': runs[0]['n_inputs'],
            'n_params': runs[0]['n_params'],
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'l2re_all': l2res,
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'theta_mean': np.mean([r['theta_final_mean'] for r in runs])
        })
    
    # Sort by L2RE for plotting
    stats_sorted = sorted(stats, key=lambda x: x['l2re_mean'])
    sorted_names = [s['name'] for s in stats_sorted]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Input Features Comparison', 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(set_names)))
    color_map = {s['name']: colors[i] for i, s in enumerate(stats_sorted)}
    
    # 1. L2RE comparison (sorted)
    ax1 = axes[0, 0]
    x_pos = np.arange(len(sorted_names))
    l2re_means = [s['l2re_mean'] for s in stats_sorted]
    l2re_stds = [s['l2re_std'] for s in stats_sorted]
    
    bars1 = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=5,
                    color=[color_map[n] for n in sorted_names], 
                    alpha=0.8, edgecolor='black')
    
    # Highlight best
    bars1[0].set_edgecolor('green')
    bars1[0].set_linewidth(3)
    
    # Mark default
    if 'default' in sorted_names:
        default_idx = sorted_names.index('default')
        bars1[default_idx].set_hatch('//')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE by Feature Set (sorted)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add n_inputs labels
    for i, s in enumerate(stats_sorted):
        ax1.text(i, l2re_means[i] * 1.2, f'n={s["n_inputs"]}', 
                ha='center', fontsize=8)
    
    # 2. L2RE vs number of inputs
    ax2 = axes[0, 1]
    
    for s in stats:
        ax2.errorbar(s['n_inputs'], s['l2re_mean'], yerr=s['l2re_std'],
                    fmt='o', markersize=10, capsize=5,
                    color=color_map[s['name']], label=s['name'])
    
    ax2.set_xlabel('Number of Input Features')
    ax2.set_ylabel('L2 Relative Error')
    ax2.set_title('L2RE vs Input Count')
    ax2.set_yscale('log')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training time comparison
    ax3 = axes[1, 0]
    time_means = [s['time_mean'] for s in stats_sorted]
    time_stds = [s['time_std'] for s in stats_sorted]
    
    bars3 = ax3.bar(x_pos, time_means, yerr=time_stds, capsize=5,
                    color=[color_map[n] for n in sorted_names],
                    alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('Training Time by Feature Set')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Loss curves (example run)
    ax4 = axes[1, 1]
    
    for name in set_names:
        if results[name] and results[name][0].get('history'):
            hist = results[name][0]['history']
            if 'losses' in hist and hist['losses']:
                ax4.semilogy(hist['losses'], label=name, 
                            color=color_map[name], linewidth=1.5, alpha=0.8)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('Training Loss (Example Run)')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_loss_weighting_comparison(results: Dict, strategies: List[str],
                                    pde_type: str, save_path: str = None):
    """
    Plot comparison of different loss weighting strategies.
    
    6 subplots:
    1. L2RE comparison (bar chart, sorted)
    2. Training loss curves
    3. Weight evolution (example run)
    4. Individual loss components
    5. L2RE box plot
    6. Final theta comparison
    """
    # Collect statistics
    stats = []
    for strategy in strategies:
        runs = results[strategy]
        l2res = [r['l2re'] for r in runs]
        times = [r['training_time'] for r in runs]
        thetas = [r['theta_final_mean'] for r in runs]
        
        stats.append({
            'strategy': strategy,
            'l2re_mean': np.mean(l2res),
            'l2re_std': np.std(l2res),
            'l2re_all': l2res,
            'time_mean': np.mean(times),
            'theta_mean': np.mean(thetas)
        })
    
    # Sort by L2RE
    stats_sorted = sorted(stats, key=lambda x: x['l2re_mean'])
    sorted_strategies = [s['strategy'] for s in stats_sorted]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Loss Weighting Strategy Comparison',
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    color_map = {s: colors[i] for i, s in enumerate(strategies)}
    
    # 1. L2RE comparison (sorted)
    ax1 = axes[0, 0]
    x_pos = np.arange(len(sorted_strategies))
    l2re_means = [s['l2re_mean'] for s in stats_sorted]
    l2re_stds = [s['l2re_std'] for s in stats_sorted]
    
    bars1 = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=5,
                    color=[color_map[s] for s in sorted_strategies],
                    alpha=0.8, edgecolor='black')
    
    bars1[0].set_edgecolor('green')
    bars1[0].set_linewidth(3)
    
    # Mark fixed baseline
    if 'fixed' in sorted_strategies:
        fixed_idx = sorted_strategies.index('fixed')
        bars1[fixed_idx].set_hatch('//')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted_strategies, rotation=45, ha='right')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE by Strategy (sorted)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Training loss curves
    ax2 = axes[0, 1]
    
    for strategy in strategies:
        if results[strategy] and results[strategy][0].get('history'):
            hist = results[strategy][0]['history']
            if 'losses' in hist and hist['losses']:
                ax2.semilogy(hist['losses'], label=strategy,
                            color=color_map[strategy], linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Training Loss (Example Run)')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 3. Weight evolution (for adaptive strategies)
    ax3 = axes[0, 2]
    
    for strategy in strategies:
        if results[strategy] and results[strategy][0].get('history'):
            hist = results[strategy][0]['history']
            if 'lambda_pde' in hist and hist['lambda_pde']:
                # Plot only lambda_bc as it's most variable
                ax3.plot(hist['lambda_bc'], label=f'{strategy} (λ_bc)',
                        color=color_map[strategy], linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('λ_bc Evolution (Example Run)')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual loss components (final values)
    ax4 = axes[1, 0]
    
    pde_losses = []
    bc_losses = []
    ic_losses = []
    
    for strategy in sorted_strategies:
        hist = results[strategy][0]['history']
        pde_losses.append(hist['pde_losses'][-1] if hist['pde_losses'] else 0)
        bc_losses.append(hist['bc_losses'][-1] if hist['bc_losses'] else 0)
        ic_losses.append(hist['ic_losses'][-1] if hist['ic_losses'] else 0)
    
    width = 0.25
    x = np.arange(len(sorted_strategies))
    
    ax4.bar(x - width, pde_losses, width, label='PDE', color='blue', alpha=0.7)
    ax4.bar(x, bc_losses, width, label='BC', color='green', alpha=0.7)
    ax4.bar(x + width, ic_losses, width, label='IC', color='red', alpha=0.7)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(sorted_strategies, rotation=45, ha='right')
    ax4.set_ylabel('Final Loss Value')
    ax4.set_title('Final Loss Components')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. L2RE box plot
    ax5 = axes[1, 1]
    
    l2re_data = [next(s for s in stats if s['strategy'] == strat)['l2re_all'] 
                 for strat in sorted_strategies]
    bp = ax5.boxplot(l2re_data, labels=sorted_strategies, patch_artist=True)
    
    for i, (patch, strat) in enumerate(zip(bp['boxes'], sorted_strategies)):
        patch.set_facecolor(color_map[strat])
        patch.set_alpha(0.7)
    
    bp['boxes'][0].set_edgecolor('green')
    bp['boxes'][0].set_linewidth(3)
    
    ax5.set_ylabel('L2 Relative Error')
    ax5.set_title('L2RE Distribution')
    ax5.set_yscale('log')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Final theta comparison
    ax6 = axes[1, 2]
    
    theta_means = [next(s for s in stats if s['strategy'] == strat)['theta_mean']
                   for strat in sorted_strategies]
    
    bars6 = ax6.bar(x_pos, theta_means, color=[color_map[s] for s in sorted_strategies],
                    alpha=0.8, edgecolor='black')
    
    ax6.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Trapezoidal')
    ax6.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit')
    ax6.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit')
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(sorted_strategies, rotation=45, ha='right')
    ax6.set_ylabel('Final θ (mean)')
    ax6.set_title('Learned θ by Strategy')
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig

def plot_model_comparison(results: Dict, pde_type: str, save_path: str = None):
    """
    Plot model comparison results.
    
    4 subplots:
    1. L2RE comparison (bar chart)
    2. Efficiency: L2RE vs Time (scatter)
    3. Loss curves (example run)
    4. Params vs L2RE
    """
    model_names = list(results.keys())
    
    # Collect statistics
    stats = []
    for name in model_names:
        runs = results[name]
        if not runs:
            continue
        
        stats.append({
            'name': name,
            'l2re_mean': np.mean([r['l2re'] for r in runs]),
            'l2re_std': np.std([r['l2re'] for r in runs]),
            'l2re_all': [r['l2re'] for r in runs],
            'rmse_mean': np.mean([r['rmse'] for r in runs]),
            'time_mean': np.mean([r['training_time'] for r in runs]),
            'time_std': np.std([r['training_time'] for r in runs]),
            'n_params': runs[0].get('n_params', 0),
            'model_type': runs[0].get('model_type', 'unknown'),
            'use_data': runs[0].get('use_data', False),
            'correction_mode': runs[0].get('correction_mode', None),
        })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Model Comparison',
                 fontsize=14, fontweight='bold')
    
    # Colors
    _CORR_COLORS = {
        None: 'lightblue', 'N/A': 'lightblue',
        'none': 'royalblue',
        'per_step_bias': '#2196F3', 'per_step_gate': '#FF9800',
        'time_bias': '#00BCD4', 'time_gate': '#FFC107',
        'rhs_scale': '#4CAF50', 'output_bias': '#E91E63',
        'step_bias+output_bias': '#3F51B5', 'time_bias+output_bias': '#673AB7',
        'time_gate+rhs_scale': '#FF5722',
    }
    
    colors = []
    for s in stats:
        if s['model_type'] == 'classical':
            colors.append('forestgreen')
        elif not s['use_data']:
            colors.append('lightblue')
        else:
            colors.append(_CORR_COLORS.get(s['correction_mode'], 'royalblue'))
    
    names = [s['name'] for s in stats]
    x_pos = np.arange(len(names))
    
    # 1. L2RE comparison
    ax1 = axes[0, 0]
    l2re_means = [s['l2re_mean'] for s in stats]
    l2re_stds = [s['l2re_std'] for s in stats]
    
    bars = ax1.bar(x_pos, l2re_means, yerr=l2re_stds, capsize=4,
                   color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    best_idx = np.argmin(l2re_means)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE Comparison')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (m, s) in enumerate(zip(l2re_means, l2re_stds)):
        ax1.text(i, m * 1.5, f'{m:.2e}', ha='center', va='bottom', fontsize=7)
    
    # 2. Efficiency: L2RE vs Time
    ax2 = axes[0, 1]
    
    for i, s in enumerate(stats):
        marker = 'o' if s['model_type'] == 'lf_pinn' else 's'
        ax2.scatter(s['time_mean'], s['l2re_mean'],
                   s=150, marker=marker, color=colors[i],
                   edgecolors='black', linewidth=1.5, label=s['name'], zorder=3)
        ax2.errorbar(s['time_mean'], s['l2re_mean'],
                    xerr=s['time_std'], yerr=s['l2re_std'],
                    fmt='none', color=colors[i], alpha=0.5, capsize=3)
    
    ax2.set_xlabel('Training Time (s)')
    ax2.set_ylabel('L2 Relative Error')
    ax2.set_title('Efficiency: L2RE vs Time (↙ better)')
    ax2.set_yscale('log')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss curves
    ax3 = axes[1, 0]
    
    for i, name in enumerate(model_names):
        if results[name] and results[name][0].get('history'):
            hist = results[name][0]['history']
            if 'losses' in hist and hist['losses']:
                ax3.semilogy(hist['losses'], label=name,
                            color=colors[i] if i < len(colors) else 'gray',
                            linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Total Loss')
    ax3.set_title('Training Loss (Example Run)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. Params vs L2RE
    ax4 = axes[1, 1]
    
    for i, s in enumerate(stats):
        marker = 'o' if s['model_type'] == 'lf_pinn' else 's'
        ax4.scatter(s['n_params'], s['l2re_mean'],
                   s=150, marker=marker, color=colors[i],
                   edgecolors='black', linewidth=1.5, label=s['name'], zorder=3)
        ax4.errorbar(s['n_params'], s['l2re_mean'], yerr=s['l2re_std'],
                    fmt='none', color=colors[i], alpha=0.5, capsize=3)
    
    ax4.set_xlabel('Number of Parameters')
    ax4.set_ylabel('L2 Relative Error')
    ax4.set_title('Params vs Accuracy')
    ax4.set_yscale('log')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


def plot_efficiency_analysis(results: Dict, pde_type: str, save_path: str = None):
    """
    Detailed efficiency analysis.
    
    4 subplots:
    1. Error per Second (L2RE / time)
    2. Error per Parameter (L2RE / params)
    3. Time-Accuracy Efficiency (improvement% / finetune_time)
    4. Combined scatter: Time vs L2RE, size ∝ params
    """
    stats = []
    colors = []
    
    _CORR_COLORS = {
        None: 'lightblue', 'N/A': 'lightblue',
        'none': 'royalblue',
        'per_step_bias': '#2196F3', 'per_step_gate': '#FF9800',
        'time_bias': '#00BCD4', 'time_gate': '#FFC107',
        'rhs_scale': '#4CAF50', 'output_bias': '#E91E63',
        'step_bias+output_bias': '#3F51B5', 'time_bias+output_bias': '#673AB7',
        'time_gate+rhs_scale': '#FF5722',
    }
    
    for name, runs in results.items():
        if not runs:
            continue
        
        s = {
            'name': name,
            'l2re_mean': np.mean([r['l2re'] for r in runs]),
            'time_mean': np.mean([r['training_time'] for r in runs]),
            'n_params': runs[0].get('n_params', 1),
            'model_type': runs[0].get('model_type', 'unknown'),
            'use_data': runs[0].get('use_data', False),
            'correction_mode': runs[0].get('correction_mode', None),
            'finetune_time_mean': np.mean([r.get('finetune_time', r['training_time']) for r in runs]),
        }
        stats.append(s)
        
        if s['model_type'] == 'classical':
            colors.append('forestgreen')
        elif not s['use_data']:
            colors.append('lightblue')
        else:
            colors.append(_CORR_COLORS.get(s['correction_mode'], 'royalblue'))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'{pde_type.replace("_", " ").title()}: Efficiency Analysis',
                 fontsize=14, fontweight='bold')
    
    names = [s['name'] for s in stats]
    x_pos = np.arange(len(names))
    
    # 1. L2RE / Time (lower = better)
    ax1 = axes[0, 0]
    efficiency_time = [s['l2re_mean'] / max(s['time_mean'], 0.1) * 100 for s in stats]
    
    bars1 = ax1.bar(x_pos, efficiency_time, color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax1.set_ylabel('L2RE / Time × 100 (↓ better)')
    ax1.set_title('Error per Second')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. L2RE / Params (lower = better)
    ax2 = axes[0, 1]
    efficiency_params = [s['l2re_mean'] / max(s['n_params'], 1) * 1000 for s in stats]
    
    bars2 = ax2.bar(x_pos, efficiency_params, color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    ax2.set_ylabel('L2RE / Params × 1000 (↓ better)')
    ax2.set_title('Error per Parameter')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Time-Accuracy Efficiency: improvement% / finetune_time
    ax3 = axes[1, 0]
    
    # Baseline L2RE = model without data (or worst L2RE)
    no_data_l2re = None
    for s in stats:
        if not s['use_data']:
            no_data_l2re = s['l2re_mean']
            break
    if no_data_l2re is None:
        no_data_l2re = max(s['l2re_mean'] for s in stats)
    
    efficiencies = []
    eff_names = []
    eff_colors = []
    for i, s in enumerate(stats):
        if s['use_data']:
            improvement = (no_data_l2re - s['l2re_mean']) / no_data_l2re * 100
            ft_time = max(s['finetune_time_mean'], 0.1)
            eff = improvement / ft_time  # %/s
            efficiencies.append(eff)
            eff_names.append(s['name'])
            eff_colors.append(colors[i])
    
    if efficiencies:
        x_eff = np.arange(len(efficiencies))
        bar_colors_eff = ['#4CAF50' if e > 0 else '#F44336' for e in efficiencies]
        bars3 = ax3.bar(x_eff, efficiencies, color=eff_colors, edgecolor='white', linewidth=0.5, alpha=0.8)
        
        for j, (bar, eff) in enumerate(zip(bars3, efficiencies)):
            va = 'bottom' if eff >= 0 else 'top'
            offset = max(abs(eff) * 0.05, 0.1)
            y_pos = eff + offset if eff >= 0 else eff - offset
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{eff:.1f}', ha='center', va=va, fontsize=8, fontweight='bold')
        
        ax3.axhline(y=0, color='black', linewidth=0.8)
        ax3.set_xticks(x_eff)
        ax3.set_xticklabels(eff_names, rotation=40, ha='right', fontsize=8)
    
    ax3.set_ylabel('Improvement% / Finetune Time (↑ better)')
    ax3.set_title('Time-Accuracy Efficiency (%/s)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Combined: Time vs L2RE with size = params
    ax4 = axes[1, 1]
    
    for i, s in enumerate(stats):
        marker = 'o' if s['model_type'] == 'lf_pinn' else 's'
        size = max(50, s['n_params'] * 2)
        ax4.scatter(s['time_mean'], s['l2re_mean'],
                   s=size, marker=marker, color=colors[i],
                   alpha=0.7, edgecolors='black', linewidths=2,
                   label=f"{s['name']} ({s['n_params']}p)")
    
    ax4.set_xlabel('Training Time (s)')
    ax4.set_ylabel('L2 Relative Error')
    ax4.set_title('Time vs Accuracy (size ∝ params)')
    ax4.set_yscale('log')
    ax4.legend(fontsize=6, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig

def plot_l2re_study(results: Dict, save_path: str = None):
    """
    Plot L2RE study results.
    
    6 subplots:
    1. L2RE vs epochs (mean ± std)
    2. L2RE drop rate per epoch
    3. Relative L2RE drop (%)
    4. Theta evolution
    5. All runs L2RE
    6. Loss vs epochs
    """
    pde_type = results['pde_type']
    epochs = results['epochs']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{TITLES[pde_type]}: PDE Training L2RE Analysis', fontsize=14, fontweight='bold')
    
    # 1. L2RE vs epochs
    ax1 = axes[0, 0]
    ax1.semilogy(epochs, results['l2re_mean'], 'b-', linewidth=2, label='Mean')
    ax1.fill_between(epochs, 
                     results['l2re_mean'] - results['l2re_std'],
                     results['l2re_mean'] + results['l2re_std'],
                     alpha=0.3, color='blue', label='±σ')
    ax1.axvline(results['min_l2re_epoch'], color='red', linestyle='--', alpha=0.7,
                label=f'Min at {results["min_l2re_epoch"]}')
    
    # Mark threshold epochs
    colors = {'90%': 'green', '95%': 'orange', '99%': 'purple'}
    for name, ep in results['threshold_epochs'].items():
        if ep is not None:
            ax1.axvline(ep, color=colors[name], linestyle=':', alpha=0.7, label=f'{name}: ep {ep}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('L2 Relative Error')
    ax1.set_title('L2RE vs Epochs')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. L2RE drop rate
    ax2 = axes[0, 1]
    drop_epochs = epochs[1:]
    ax2.plot(drop_epochs, results['l2re_drop'], 'r-', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(results['best_drop_epoch'], color='green', linestyle='--', 
                label=f'Steepest at {results["best_drop_epoch"]}')
    ax2.scatter([results['best_drop_epoch']], 
                [results['l2re_drop'][list(epochs).index(results['best_drop_epoch']) - 1]],
                s=100, c='green', zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L2RE Drop (negative = improvement)')
    ax2.set_title('Absolute L2RE Drop')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Relative L2RE drop
    ax3 = axes[0, 2]
    ax3.plot(drop_epochs, results['l2re_relative_drop'], 'purple', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax3.axvline(results['best_relative_drop_epoch'], color='green', linestyle='--',
                label=f'Steepest at {results["best_relative_drop_epoch"]}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Relative Drop (%)')
    ax3.set_title('Relative L2RE Drop (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Theta evolution
    ax4 = axes[1, 0]
    ax4.plot(epochs, results['theta_mean'], 'r-', linewidth=2, label='θ mean')
    ax4.fill_between(epochs,
                     results['theta_mean'] - results['theta_std'],
                     results['theta_mean'] + results['theta_std'],
                     alpha=0.3, color='red')
    ax4.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Trapezoidal (0.5)')
    ax4.axhline(0.0, color='blue', linestyle='--', alpha=0.5, label='Implicit (0)')
    ax4.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='Explicit (1)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('θ')
    ax4.set_title('Theta Evolution')
    ax4.legend(fontsize=8)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    
    # 5. All runs L2RE
    ax5 = axes[1, 1]
    for i, (run_epochs, run_l2re) in enumerate(zip(results['all_epochs'], results['all_l2re'])):
        ax5.semilogy(run_epochs, run_l2re, alpha=0.5, linewidth=1, label=f'Run {i+1}')
    ax5.semilogy(epochs, results['l2re_mean'], 'k-', linewidth=3, label='Mean')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('L2 Relative Error')
    ax5.set_title(f'All {results["n_runs"]} Runs')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # 6. Loss vs epochs
    ax6 = axes[1, 2]
    valid_loss = [(e, l) for e, l in zip(epochs, results['loss_mean']) if not np.isnan(l)]
    if valid_loss:
        loss_epochs, loss_vals = zip(*valid_loss)
        ax6.semilogy(loss_epochs, loss_vals, 'b-', linewidth=2)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Total Loss')
    ax6.set_title('Training Loss')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


MODE_CONFIG = {
    'none':           {'label': 'No correction\n(baseline)',  'color': '#9E9E9E'},
    'per_step_bias':  {'label': 'Per-step bias\n(δᵢ → θ+δ)',  'color': '#2196F3'},
    'per_step_gate':  {'label': 'Per-step gate\n(mix θ & μ)', 'color': '#FF9800'},
    'time_bias':      {'label': 'Time bias\n(interp δ(t))',   'color': '#00BCD4'},
    'time_gate':      {'label': 'Time gate\n(interp g,μ(t))', 'color': '#FFC107'},
    'rhs_scale':      {'label': 'RHS scale\n(eᵉⁱ · update)',  'color': '#4CAF50'},
    'output_bias':    {'label': 'Output bias\n(y + bᵢ)',      'color': '#E91E63'},
    'per_step_affine':{'label': 'Per-step affine\n(a·θ+b)',   'color': '#9C27B0'},
    'shared_bias':    {'label': 'Shared bias\n(one δ)',        'color': '#795548'},
}


"""
Визуализация результатов сравнения параметрических коррекций theta.

Использование:
    from tests.test_theta_params import test_theta_param_modes
    from utils.plot_theta_comparison import plot_correction_comparison
    
    results = test_theta_param_modes('heat', n_runs=5, ...)
    plot_correction_comparison(results, pde_type='heat')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11


# Короткие подписи и цвета
MODE_CONFIG = {
    'none':               {'label': 'Baseline',           'color': '#9E9E9E'},
    'per_step_bias':      {'label': 'Step Bias',          'color': '#2196F3'},
    'per_step_gate':      {'label': 'Step Gate',          'color': '#FF9800'},
    'per_step_affine':    {'label': 'Step Affine',        'color': '#9C27B0'},
    'shared_bias':        {'label': 'Shared Bias',        'color': '#795548'},
    'time_bias':          {'label': 'Time Bias',          'color': '#00BCD4'},
    'time_gate':          {'label': 'Time Gate',          'color': '#FFC107'},
    'rhs_scale':          {'label': 'RHS Scale',          'color': '#4CAF50'},
    'output_bias':        {'label': 'Output Bias',        'color': '#E91E63'},
    'time_bias+rhs_scale':{'label': 'T.Bias+RHS',        'color': '#3F51B5'},
    'time_gate+rhs_scale':{'label': 'T.Gate+RHS',        'color': '#FF5722'},
}


def _get_cfg(mode):
    return MODE_CONFIG.get(mode, {'label': mode, 'color': '#607D8B'})


def plot_correction_comparison(results: dict, pde_type: str = '', save_path: str = None):
    """
    Сравнительный график результатов test_theta_param_modes.
    
    Args:
        results: dict[mode] -> list of run dicts
        pde_type: название PDE (для заголовка)
        save_path: путь для сохранения (None = показать)
    """
    modes = list(results.keys())
    n_modes = len(modes)
    
    # Статистика
    pretrain_means = []
    finetune_means = []
    finetune_stds = []
    improvements = []
    n_params_list = []
    
    for mode in modes:
        runs = results[mode]
        pretrain_means.append(np.mean([r['l2re_pretrain'] for r in runs]))
        finetune_means.append(np.mean([r['l2re_finetune'] for r in runs]))
        finetune_stds.append(np.std([r['l2re_finetune'] for r in runs]))
        improvements.append(np.mean([r['improvement_pct'] for r in runs]))
        n_params_list.append(runs[0]['n_correction_params'])
    
    colors = [_get_cfg(m)['color'] for m in modes]
    labels = [_get_cfg(m)['label'] for m in modes]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle(f'Correction Methods Comparison — {pde_type.upper() if pde_type else "PDE"}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    x = np.arange(n_modes)
    
    # ─── Plot 1: Pretrain vs Finetune L2RE ───
    ax = axes[0]
    width = 0.35
    
    ax.bar(x - width/2, pretrain_means, width, 
           label='After PDE pretrain', color='#BDBDBD', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, finetune_means, width, yerr=finetune_stds,
           label='After data finetune', color=colors, edgecolor='white', linewidth=0.5,
           capsize=3, error_kw={'linewidth': 1.5})
    
    ax.set_ylabel('L2 Relative Error')
    ax.set_title('Pretrain → Finetune Error')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=35, ha='right')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # ─── Plot 2: Improvement % ───
    ax = axes[1]
    
    bar_colors = ['#4CAF50' if imp > 0 else '#F44336' for imp in improvements]
    bars = ax.bar(x, improvements, 0.6, color=bar_colors, edgecolor='white', linewidth=0.5)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        va = 'bottom' if imp >= 0 else 'top'
        offset = max(abs(imp) * 0.05, 0.5)
        y_pos = imp + offset if imp >= 0 else imp - offset
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{imp:+.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Error Reduction after Finetune')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # ─── Plot 3: Efficiency (error vs params) ───
    ax = axes[2]
    
    for i, mode in enumerate(modes):
        if mode == 'none':
            ax.axhline(y=finetune_means[i], color='#9E9E9E', linestyle='--', 
                       linewidth=1, alpha=0.7, label='Baseline')
            continue
        
        cfg = _get_cfg(mode)
        size = max(abs(improvements[i]) * 8, 50)
        ax.scatter(n_params_list[i], finetune_means[i], 
                   s=size, c=cfg['color'], edgecolors='black', linewidth=1,
                   zorder=5, label=cfg['label'])
        
        ax.annotate(f'{n_params_list[i]}p', 
                    (n_params_list[i], finetune_means[i]),
                    textcoords="offset points", xytext=(8, -5), fontsize=9)
    
    ax.set_xlabel('Correction Parameters')
    ax.set_ylabel('Finetune L2RE')
    ax.set_title('Efficiency: Error vs Parameters')
    ax.set_yscale('log')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_step_scaling_comparison(results: dict, pde_type: str = '', save_path: str = None):
    """
    График для test_step_scaling.
    """
    all_keys = list(results.keys())
    modes = sorted(set(k[0] for k in all_keys))
    ft_steps_list = sorted(set(k[1] for k in all_keys))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Step Scaling — {pde_type.upper() if pde_type else "PDE"}',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ─── Plot 1: Finetune L2RE vs n_steps ───
    ax = axes[0]
    for mode in modes:
        ft_means, ft_stds, steps = [], [], []
        for ft_s in ft_steps_list:
            key = (mode, ft_s)
            if key in results and len(results[key]) > 0:
                runs = results[key]
                ft_means.append(np.mean([r['l2re_finetune'] for r in runs]))
                ft_stds.append(np.std([r['l2re_finetune'] for r in runs]))
                steps.append(ft_s)
        
        cfg = _get_cfg(mode)
        ax.errorbar(steps, ft_means, yerr=ft_stds, 
                     marker='o', linewidth=2, markersize=8, capsize=4,
                     label=cfg['label'], color=cfg['color'])
    
    ax.set_xlabel('Finetune n_steps')
    ax.set_ylabel('Finetune L2RE')
    ax.set_title('Error vs Number of Steps')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    
    # ─── Plot 2: Improvement % vs n_steps ───
    ax = axes[1]
    for mode in modes:
        imps, steps = [], []
        for ft_s in ft_steps_list:
            key = (mode, ft_s)
            if key in results and len(results[key]) > 0:
                runs = results[key]
                imps.append(np.mean([r['improvement_pct'] for r in runs]))
                steps.append(ft_s)
        
        cfg = _get_cfg(mode)
        ax.plot(steps, imps, marker='s', linewidth=2, markersize=8,
                label=cfg['label'], color=cfg['color'])
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Finetune n_steps')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Error Reduction vs Number of Steps')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig