import numpy as np
import matplotlib.pyplot as plt
import torch

def plot(results, history, title):
    """Простая визуализация результатов"""
    points = results['points']
    u_pred = results['u_pred']
    u_exact = results.get('u_exact', None)
    grid_shape = results.get('grid_shape', None)
    is_spatial_2d = results.get('is_spatial_2d', False)
    
    is_2d = points.shape[1] == 3
    has_grid = grid_shape is not None
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)
    
    # График 1: Решение PINN
    ax1 = axes[0, 0]
    if is_2d and has_grid:
        # 2D тепловая карта
        vals = u_pred.detach().cpu().numpy().reshape(grid_shape)
        im = ax1.imshow(vals, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax1)
        ax1.set_title('PINN решение')
    elif not is_2d and has_grid:
        # 1D+время или 1D+y: 3D поверхность
        ax1.remove()
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        x_vals = points[:, 0].detach().cpu().numpy().reshape(grid_shape)
        second_vals = points[:, 1].detach().cpu().numpy().reshape(grid_shape)
        u_vals = u_pred.detach().cpu().numpy().reshape(grid_shape)
        ax1.plot_surface(x_vals, second_vals, u_vals, cmap='viridis', alpha=0.8)
        
        # Подписи осей
        ax1.set_xlabel('x')
        if is_spatial_2d:
            ax1.set_ylabel('y')
        else:
            ax1.set_ylabel('t')
        ax1.set_zlabel('u')
        ax1.set_title('PINN решение')
    else:
        # Простой scatter
        x_vals = points[:, 0].detach().cpu().numpy()
        ax1.scatter(x_vals, u_pred.detach().cpu().numpy(), alpha=0.7, s=20)
        ax1.set_xlabel('x'), ax1.set_ylabel('u')
        ax1.set_title('PINN решение'), ax1.grid(True)
    
    # График 2: История потерь
    ax2 = axes[0, 1]
    ax2.semilogy(history['losses'])
    ax2.set_xlabel('Эпоха'), ax2.set_ylabel('Потеря')
    ax2.set_title('Обучение'), ax2.grid(True)
    
    # График 3: Точное решение
    ax3 = axes[0, 2]
    if u_exact is not None:
        if is_2d and has_grid:
            vals = u_exact.detach().cpu().numpy().reshape(grid_shape)
            im = ax3.imshow(vals, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax3)
            ax3.set_title('Точное решение')
        elif not is_2d and has_grid:
            ax3.remove()
            ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            u_exact_vals = u_exact.detach().cpu().numpy().reshape(grid_shape)
            ax3.plot_surface(x_vals, second_vals, u_exact_vals, cmap='viridis', alpha=0.8)
            
            # Правильные подписи осей
            ax3.set_xlabel('x')
            if is_spatial_2d:
                ax3.set_ylabel('y')
            else:
                ax3.set_ylabel('t')
            ax3.set_zlabel('u')
            ax3.set_title('Точное решение')
        else:
            x_vals = points[:, 0].detach().cpu().numpy()
            ax3.scatter(x_vals, u_exact.detach().cpu().numpy(), alpha=0.7, s=20)
            ax3.set_xlabel('x'), ax3.set_ylabel('u')
            ax3.set_title('Точное решение'), ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Нет точного\nрешения', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Точное решение')
    
    # График 4: Параметр u
    ax4 = axes[1, 0]
    ax4.plot(history['u_values'])
    ax4.axhline(0.5, color='r', linestyle='--', alpha=0.7, label='u=0.5')
    ax4.set_xlabel('Эпоха'), ax4.set_ylabel('u')
    ax4.set_title('Параметр трапеций'), ax4.grid(True), ax4.legend()
    
    # График 5: Физические параметры
    ax5 = axes[1, 1]
    if history['params']:
        params = history['params'][0]
        for key in params:
            if key != 'u':
                vals = [p[key] for p in history['params']]
                ax5.plot(vals, label=key)
        ax5.set_xlabel('Эпоха'), ax5.set_ylabel('Значение')
        ax5.set_title('Физические параметры'), ax5.grid(True)
        if len([k for k in params if k != 'u']) > 0:
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Нет параметров', ha='center', va='center',
                    transform=ax5.transAxes)
    
    # График 6: Сравнение LF-PINN и точного решения
    ax6 = axes[1, 2]
    if u_exact is not None and has_grid and not is_spatial_2d:
        # Только для эволюционных уравнений (с временем)
        x_vals = points[:, 0].detach().cpu().numpy().reshape(grid_shape)
        t_vals = points[:, 1].detach().cpu().numpy().reshape(grid_shape)
        u_pred_vals = u_pred.detach().cpu().numpy().reshape(grid_shape)
        u_exact_vals = u_exact.detach().cpu().numpy().reshape(grid_shape)

        t_min = t_vals[0, :].min()
        t_max = t_vals[0, :].max()
        n_slices = 5
        times_to_plot = np.linspace(t_min, t_max, n_slices)
        
        colors = ['b', 'g', 'r', 'c', 'm']

        for i, t0 in enumerate(times_to_plot):
            idx = np.argmin(np.abs(t_vals[0, :] - t0))
            ax6.plot(x_vals[:, idx], u_exact_vals[:, idx],
                    color=colors[i], linestyle='-', linewidth=2,
                    label=f'Точное t={t0:.2f}')
            ax6.plot(x_vals[:, idx], u_pred_vals[:, idx],
                    color=colors[i], linestyle='--', linewidth=2,
                    label=f'PINN t={t0:.2f}')

        ax6.set_xlabel('x'), ax6.set_ylabel('u')
        ax6.set_title('Сравнение PINN и точного решения')
        ax6.legend(fontsize=8), ax6.grid(True)
    else:
        # Для стационарных уравнений (Пуассон)
        ax6.text(0.5, 0.5, 'Стационарное\nуравнение\n(нет эволюции)',
                 ha='center', va='center',
                 transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Сравнение')
    
    plt.tight_layout()
    plt.show()
    
    # Финальные параметры
    if history['params']:
        final = history['params'][-1]
        print(f"Финальные параметры: {final}")
        print(f"Финальная потеря: {history['losses'][-1]:.4e}")