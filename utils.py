import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch


def simulate_gp_phs(simulator, dEdp_model, dEdq_model):
    print("Executing GP-PHS trajectory rollout simulation...")
    q0 = -4.0 * (simulator.z - simulator.L / 2) * np.exp(-(simulator.z - simulator.L / 2) ** 2)
    p0 = np.zeros(simulator.N)
    state0 = np.concatenate([p0, q0])

    dEdp_model.eval()
    dEdq_model.eval()

    def gp_pde_rhs(t, state):
        p = state[:simulator.N]
        q = state[simulator.N:]

        # NEW: Include simulator.z in the evaluation tensor
        X_raw = torch.tensor(np.stack([p, q, simulator.z], axis=1), dtype=torch.float32)

        # Normalize inputs before passing to GP
        X_norm = (X_raw - dEdp_model.X_mean) / (dEdp_model.X_std + 1e-6)

        with torch.no_grad():
            pred_p_norm = dEdp_model(X_norm).mean.numpy()
            pred_q_norm = dEdq_model(X_norm).mean.numpy()

        # Un-normalize outputs back into physical parameter scales
        grad_H_p = pred_p_norm * dEdp_model.y_std.numpy() + dEdp_model.y_mean.numpy()
        grad_H_q = pred_q_norm * dEdq_model.y_std.numpy() + dEdq_model.y_mean.numpy()

        return simulator.evaluate_ph_dynamics(p, q, grad_H_p, grad_H_q, t)

    t_eval = np.arange(0, simulator.T, simulator.dt)
    sol = solve_ivp(gp_pde_rhs, [0, simulator.T], state0, t_eval=t_eval, method='RK45')

    return sol.y[:simulator.N, :], sol.y[simulator.N:, :]


def plot_comparison(data, dEdp_model, dEdq_model, simulator, save_path="gpphs_results.png"):
    p_min, p_max = np.min(data['p']), np.max(data['p'])
    q_min, q_max = np.min(data['q']), np.max(data['q'])

    p_grid, q_grid = np.meshgrid(np.linspace(p_min, p_max, 40), np.linspace(q_min, q_max, 40))
    p_test_flat = p_grid.flatten()
    q_test_flat = q_grid.flatten()

    # NEW: Pick a fixed z location to visualize the surface and create an array of it
    mid_z_idx = int(simulator.N / 2)
    z_val = simulator.z[mid_z_idx]
    z_test_flat = np.full_like(p_test_flat, z_val)

    real_dEdp = simulator.dEdp(p_test_flat)
    real_dEdq = np.array([simulator.dEdq(q, mid_z_idx) for q in q_test_flat])

    # NEW: Include z_test_flat in the test tensor
    X_test_raw = torch.tensor(np.stack([p_test_flat, q_test_flat, z_test_flat], axis=1), dtype=torch.float32)

    X_test_norm = (X_test_raw - dEdp_model.X_mean) / (dEdp_model.X_std + 1e-6)

    dEdp_model.eval()
    dEdq_model.eval()

    with torch.no_grad():
        pred_dEdp_norm = dEdp_model(X_test_norm).mean.numpy()
        pred_dEdq_norm = dEdq_model(X_test_norm).mean.numpy()

    pred_dEdp = pred_dEdp_norm * dEdp_model.y_std.numpy() + dEdp_model.y_mean.numpy()
    pred_dEdq = pred_dEdq_norm * dEdq_model.y_std.numpy() + dEdq_model.y_mean.numpy()

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(p_grid, q_grid, real_dEdp.reshape(40, 40), cmap='viridis', alpha=0.5, edgecolor='none')
    ax1.scatter(p_test_flat[::4], q_test_flat[::4], pred_dEdp[::4], color='red', label='GP Prediction', s=12)
    ax1.set_title('dH/dp (Kinetic Energy Gradient Profile)', fontsize=12)
    ax1.set_xlabel('p')
    ax1.set_ylabel('q')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(p_grid, q_grid, real_dEdq.reshape(40, 40), cmap='plasma', alpha=0.5, edgecolor='none')
    ax2.scatter(p_test_flat[::4], q_test_flat[::4], pred_dEdq[::4], color='blue', label='GP Prediction', s=12)
    ax2.set_title(f'dH/dq (Strain Energy Gradient at z={z_val:.2f})', fontsize=12)
    ax2.set_xlabel('p')
    ax2.set_ylabel('q')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Graphical evaluation saved as '{save_path}'")


def plot_time_slices(data, q_gp, simulator, time_slices, save_path="gpphs_string_slices.png"):
    """
    Plots the spatial shape of the 1D string (displacement u) at specific time steps
    comparing the True Numerical Solution vs the GP-PHS Rollout.
    """
    q_true = data['q']
    z = simulator.z
    dz = simulator.dz

    # Calculate displacement u(z) by integrating the strain q spatially
    u_true = np.cumsum(q_true, axis=0) * dz
    u_gp = np.cumsum(q_gp, axis=0) * dz

    fig, axs = plt.subplots(1, len(time_slices), figsize=(18, 5), sharey=True)

    for idx, step in enumerate(time_slices):
        if step >= u_true.shape[1] or step >= u_gp.shape[1]:
            print(f"[WARNING] Time step {step} is out of bounds for current simulation length.")
            continue

        time_val = step * simulator.dt

        axs[idx].plot(z, u_true[:, step], 'k-', linewidth=2.5, label='True Numerical PDE')
        axs[idx].plot(z, u_gp[:, step], 'r--', linewidth=2.5, label='GP-PHS Rollout')

        axs[idx].set_title(f'Time Step = {step} ({time_val:.2f}s)', fontsize=12)
        axs[idx].set_xlabel('Spatial Coordinate (z)', fontsize=10)
        if idx == 0:
            axs[idx].set_ylabel('Displacement Profile (u)', fontsize=11)
        axs[idx].grid(True, linestyle=':', alpha=0.6)
        axs[idx].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] 1D String time-slice comparisons saved as '{save_path}'")
