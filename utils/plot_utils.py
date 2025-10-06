import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

# ============================================================
# ОСНОВНАЯ ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot(results, history, title="PINN Solution"):
    points = results['points']
    u_pred = results['u_pred'].detach().numpy()
    u_exact = results.get('u_exact', None)
    grid_shape = results['grid_shape']
    is_spatial_2d = results['is_spatial_2d']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Предсказание PINN
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
    
    # 2. История обучения
    ax2 = axes[0, 1]
    epochs = range(len(history['losses']))
    ax2.semilogy(epochs, history['losses'], 'b-', linewidth=2, label='Total', alpha=0.8)
    ax2.semilogy(epochs, history['pde_losses'], 'r-', linewidth=1.5, label='PDE', alpha=0.7)
    ax2.semilogy(epochs, history['bc_losses'], 'g-', linewidth=1.5, label='BC', alpha=0.7)
    if any(ic > 0 for ic in history['ic_losses']):
        ax2.semilogy(epochs, history['ic_losses'], 'm-', linewidth=1.5, label='IC', alpha=0.7)
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Loss (log)')
    ax2.set_title('Training History'), ax2.legend(), ax2.grid(True, alpha=0.3)
    
    # 3. Точное решение
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
    
    # 4. Параметр θ(x,t)
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
    
    # 5. Ошибка
    ax5 = axes[1, 1]
    if u_exact is not None and 'error' in results:
        error = results['error'].detach().numpy()
        if is_spatial_2d:
            err = error.reshape(grid_shape)
            im5 = ax5.contourf(x, y, err, levels=50, cmap='hot')
            ax5.set_xlabel('x'), ax5.set_ylabel('y')
            ax5.set_title(f'Error (max={results["max_error"]:.2e})')
            plt.colorbar(im5, ax=ax5)
        else:
            err = error.reshape(grid_shape)
            im5 = ax5.contourf(x, t, err, levels=50, cmap='hot')
            ax5.set_xlabel('x'), ax5.set_ylabel('t')
            ax5.set_title(f'Error (max={results["max_error"]:.2e})')
            plt.colorbar(im5, ax=ax5)
    else:
        ax5.axis('off')
    
    # 6. Срезы по времени
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
            print(f"Max Error:  {results['max_error']:.4e}")
        print(f"{'='*60}\n")

# ============================================================
# СРАВНЕНИЕ МЕТОДОВ
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
    
    # Потери
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
# ПОЛЕ THETA
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
        theta_values = model.theta_net(torch.cat([x_flat, t_flat, t_next], dim=1))
    
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