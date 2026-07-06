import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
import gpytorch


def stratified_z_indices(
    X,
    train_fraction,
    seed=42,
    z_col=2,
    left_region=(0.0, 2.0),
    middle_region=(2.0, 8.0),
    right_region=(8.0, 10.0),
    weights=(0.30, 0.40, 0.30),
    include_domain_boundaries=True,
    shuffle=True,
):
    """
    Returns stratified training indices based on spatial coordinate z.

    Sampling allocation:
        30% from left region:   0 <= z < 2
        40% from middle region: 2 <= z < 8
        30% from right region:  8 <= z <= 10

    Args:
        X:
            Input array of shape [N_samples, 3], where X[:, z_col] is z.
        train_fraction:
            Fraction of the full dataset to sample.
        seed:
            Random seed.
        z_col:
            Column index of z in X.
        left_region, middle_region, right_region:
            Spatial intervals.
        weights:
            Sampling fractions for left, middle, right regions.
        include_domain_boundaries:
            If True, include z=0 and z=10 in left/right regions.
            If False, strictly use 0 < z < 2 and 8 < z < 10.
        shuffle:
            Whether to shuffle the final returned index array.

    Returns:
        train_idx:
            Array of selected training indices.
    """
    rng = np.random.default_rng(seed)

    z = X[:, z_col]
    num_total = X.shape[0]
    num_train = int(num_total * train_fraction)

    w_left, w_middle, w_right = weights

    n_left = int(round(num_train * w_left))
    n_middle = int(round(num_train * w_middle))
    n_right = num_train - n_left - n_middle

    z0, z1 = left_region
    z2, z3 = middle_region
    z4, z5 = right_region

    if include_domain_boundaries:
        left_mask = (z >= z0) & (z < z1)
        middle_mask = (z >= z2) & (z < z3)
        right_mask = (z >= z4) & (z <= z5)
    else:
        left_mask = (z > z0) & (z < z1)
        middle_mask = (z > z2) & (z < z3)
        right_mask = (z > z4) & (z < z5)

    left_idx_pool = np.where(left_mask)[0]
    middle_idx_pool = np.where(middle_mask)[0]
    right_idx_pool = np.where(right_mask)[0]

    def sample_from_pool(pool, n):
        if len(pool) == 0:
            return np.array([], dtype=int)
        n = min(n, len(pool))
        return rng.choice(pool, size=n, replace=False)

    left_idx = sample_from_pool(left_idx_pool, n_left)
    middle_idx = sample_from_pool(middle_idx_pool, n_middle)
    right_idx = sample_from_pool(right_idx_pool, n_right)

    train_idx = np.concatenate([left_idx, middle_idx, right_idx])

    # If some region did not have enough points, fill the remaining budget
    # from points not already selected.
    if len(train_idx) < num_train:
        selected_mask = np.zeros(num_total, dtype=bool)
        selected_mask[train_idx] = True

        remaining_pool = np.where(~selected_mask)[0]
        n_missing = num_train - len(train_idx)

        extra_idx = rng.choice(
            remaining_pool,
            size=min(n_missing, len(remaining_pool)),
            replace=False,
        )

        train_idx = np.concatenate([train_idx, extra_idx])

    if shuffle:
        rng.shuffle(train_idx)

    return train_idx


def random_z_indices(X, train_fraction, seed):
    num_train = int(X.shape[0] * train_fraction)
    rng = np.random.default_rng(seed=seed)
    return rng.choice(X.shape[0], size=num_train, replace=False)


def simulate_gp_phs(simulator, dEdp_model, dEdq_model, get_init_q_config):
    print("Executing GP-PHS trajectory rollout simulation...")
    q0 = get_init_q_config(simulator.z, simulator.L)
    p0 = np.zeros(simulator.N)
    state0 = np.concatenate([p0, q0])

    dEdp_model.eval()
    dEdq_model.eval()

    def gp_pde_rhs(t, state):
        p = state[:simulator.N]
        q = state[simulator.N:]

        # NEW: Include simulator.z in the evaluation tensor
        X_raw = torch.tensor(np.stack([p, q, simulator.z], axis=1), dtype=torch.float64)

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


def plot_comparison(data, dEdp_model, dEdq_model,
                    simulator, save_path="gpphs_results.png", legend_tag="baseline"):
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
    X_test_raw = torch.tensor(np.stack([p_test_flat, q_test_flat, z_test_flat], axis=1), dtype=torch.float64)

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
    ax1.scatter(p_test_flat[::4], q_test_flat[::4], pred_dEdp[::4], color='red', label=f'GP Prediction ({legend_tag})', s=12)
    ax1.set_title('dH/dp (Kinetic Energy Gradient Profile)', fontsize=12)
    ax1.set_xlabel('p')
    ax1.set_ylabel('q')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(p_grid, q_grid, real_dEdq.reshape(40, 40), cmap='plasma', alpha=0.5, edgecolor='none')
    ax2.scatter(p_test_flat[::4], q_test_flat[::4], pred_dEdq[::4], color='blue', label=f'GP Prediction ({legend_tag})', s=12)
    ax2.set_title(f'dH/dq (Strain Energy Gradient at z={z_val:.2f})', fontsize=12)
    ax2.set_xlabel('p')
    ax2.set_ylabel('q')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Graphical evaluation saved as '{save_path}'")


def plot_effort_force_rmse_and_qmax_over_time(
    dHdq_summary,
    q_gp,
    simulator,
    data=None,
    edge=5,
    title="Effort / Force Error and Strain Growth",
    save_path=None,
):
    """
    Plots over time:
        1. RMSE of dH/dq
        2. RMSE of d/dz(dH/dq)
        3. max_z |q_GP|

    Args:
        dHdq_summary:
            Dictionary containing flattened or grid-shaped dH/dq and force fields.
            Expected keys:
                "dHdq_mean" or "dHdq_pred"
                "dHdq_true"
                "dHdq_force_pred"
                "dHdq_force_true"
        q_gp:
            GP rollout strain, shape [N_z, N_t].
        simulator:
            Simulator object with z, dt, and N.
        data:
            Optional true data dictionary. If provided and contains data["q"],
            also plots true max_z |q|.
        grid_shape:
            Tuple (N_z, N_t). If None, inferred from q_gp.shape.
        edge:
            Number of boundary cells excluded from RMSE and max computations.
        title:
            Figure title.
        save_path:
            Optional path to save the figure.
    """
    N_z, N_t = q_gp.shape
    interior = slice(edge, N_z - edge)

    # ------------------------------------------------------------
    # Get dH/dq prediction and true values
    # ------------------------------------------------------------
    dHdq_pred = np.asarray(dHdq_summary["dHdq_mean"]).reshape(N_z, N_t)
    dHdq_true = np.asarray(dHdq_summary["dHdq_true"]).reshape(N_z, N_t)

    # ------------------------------------------------------------
    # Get force-field prediction and true values
    # ------------------------------------------------------------
    force_pred = np.asarray(dHdq_summary["dHdq_force_pred"]).reshape(N_z, N_t)
    force_true = np.asarray(dHdq_summary["dHdq_force_true"]).reshape(N_z, N_t)

    # ------------------------------------------------------------
    # Time-resolved RMSEs
    # ------------------------------------------------------------
    dHdq_err = dHdq_pred - dHdq_true
    force_err = force_pred - force_true

    dHdq_rmse_t = np.sqrt(
        np.nanmean(dHdq_err[interior, :] ** 2, axis=0)
    )

    force_rmse_t = np.sqrt(
        np.nanmean(force_err[interior, :] ** 2, axis=0)
    )

    qmax_gp_t = np.nanmax(np.abs(q_gp[interior, :]), axis=0)

    # ------------------------------------------------------------
    # Time array
    # ------------------------------------------------------------
    if data is not None and "t" in data:
        t_arr = data["t"]
    else:
        t_arr = np.arange(N_t) * simulator.dt

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(t_arr, dHdq_rmse_t)
    axs[0].set_ylabel(r"RMSE of $\partial_q H$")
    axs[0].set_title(title)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t_arr, force_rmse_t)
    axs[1].set_ylabel(r"RMSE of $\partial_z(\partial_q H)$")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(t_arr, qmax_gp_t, label=r"GP $\max_z |q|$")

    if data is not None and "q" in data:
        q_true = data["q"]
        qmax_true_t = np.nanmax(np.abs(q_true[interior, :]), axis=0)
        axs[2].plot(t_arr, qmax_true_t, "--", label=r"True $\max_z |q|$")

    axs[2].set_xlabel("t")
    axs[2].set_ylabel(r"$\max_z |q|$")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "t": t_arr,
        "dHdq_rmse_t": dHdq_rmse_t,
        "force_rmse_t": force_rmse_t,
        "qmax_gp_t": qmax_gp_t,
    }


def plot_force_field_error_heatmap(
    dHdq_summary,
    simulator,
    data=None,
    grid_shape=None,
    edge=5,
    log_scale=False,
    title="Force-Field Error Heatmap",
    save_path=None,
):
    """
    Plots heatmap of force-field error:

        | d/dz(dH/dq_pred) - d/dz(dH/dq_true) |

    Args:
        dHdq_summary:
            Dictionary containing:
                "dHdq_force_error"
            or containing:
                "dHdq_force_pred" and "dHdq_force_true"
        simulator:
            Simulator object with z and dt.
        data:
            Optional data dictionary. If data["t"] exists, uses it for time axis.
        grid_shape:
            Tuple (N_z, N_t).
        edge:
            Boundary cells to mask out.
        log_scale:
            If True, plots log10 absolute error.
        title:
            Figure title.
        save_path:
            Optional path to save figure.
    """
    if grid_shape is None:
        # Try to infer from simulator.N and force array size.
        if "dHdq_force_error" in dHdq_summary:
            total_size = np.asarray(dHdq_summary["dHdq_force_error"]).size
        else:
            total_size = np.asarray(dHdq_summary["dHdq_force_pred"]).size

        N_z = simulator.N
        N_t = total_size // N_z
    else:
        N_z, N_t = grid_shape

    # ------------------------------------------------------------
    # Get force error
    # ------------------------------------------------------------
    force_error = np.asarray(dHdq_summary["dHdq_force_error"]).reshape(N_z, N_t)
    force_error_abs = np.abs(force_error)

    # Mask boundary regions
    force_error_plot = force_error_abs.copy()
    force_error_plot[:edge, :] = np.nan
    force_error_plot[-edge:, :] = np.nan

    if log_scale:
        force_error_plot = np.log10(force_error_plot + 1e-12)
        cbar_label = r"$\log_{10}(|\Delta \partial_z(\partial_q H)| + 10^{-12})$"
    else:
        cbar_label = r"$|\Delta \partial_z(\partial_q H)|$"

    # ------------------------------------------------------------
    # Axis arrays
    # ------------------------------------------------------------
    z = simulator.z

    if data is not None and "t" in data:
        t_arr = data["t"]
    else:
        t_arr = np.arange(N_t) * simulator.dt

    extent = (t_arr[0], t_arr[-1], z[0], z[-1])

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=(9, 4.8))

    im = plt.imshow(
        force_error_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    plt.colorbar(im, label=cbar_label)
    plt.xlabel("t")
    plt.ylabel("z")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "force_error_abs": force_error_abs,
        "force_error_plot": force_error_plot,
        "t": t_arr,
        "z": z,
    }


def plot_time_slices(data, q_gp, simulator, time_slices,
                     save_path="gpphs_string_slices.png", legend_tag="baseline"):
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

    fig.suptitle("1D String Configuration (Varying Spatial stiffness)", fontsize=16, fontweight='bold')

    for idx, step in enumerate(time_slices):
        if step >= u_true.shape[1] or step >= u_gp.shape[1]:
            print(f"[WARNING] Time step {step} is out of bounds for current simulation length.")
            continue

        time_val = step * simulator.dt

        axs[idx].plot(z, u_true[:, step], 'k-', linewidth=2.5, label='True Numerical PDE')
        axs[idx].plot(z, u_gp[:, step], 'r--', linewidth=2.5, label=f'GP-PHS Prediction ({legend_tag})')

        axs[idx].set_title(f'Time Step = {step} ({time_val:.2f}s)', fontsize=12)
        axs[idx].set_xlabel('Spatial Coordinate (z)', fontsize=10)
        if idx == 0:
            axs[idx].set_ylabel('Displacement Profile (u)', fontsize=11)
        axs[idx].grid(True, linestyle=':', alpha=0.6)
        axs[idx].legend(loc='upper right')

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] 1D String time-slice comparisons saved as '{save_path}'")


def plot_average_time_slices(
    data,
    q_gp_mean,
    q_gp_std,
    simulator,
    time_slices,
    save_path="average_string_slices.png",
    legend_tag="multi-seed average",
    show_std_band=True,
):
    """
    Plot average GP-PHS rollout displacement over multiple seeds.

    Args:
        data:
            True test data dictionary containing data["q"].
        q_gp_mean:
            Mean predicted q over seeds, shape [N_z, N_t].
        q_gp_std:
            Std predicted q over seeds, shape [N_z, N_t].
        simulator:
            Simulator object.
        time_slices:
            Time steps to plot.
        save_path:
            Output path.
        legend_tag:
            Label for averaged GP prediction.
        show_std_band:
            If True, shade +/- 1 std in displacement.
    """

    q_true = data["q"]
    z = simulator.z
    dz = simulator.dz

    u_true = np.cumsum(q_true, axis=0) * dz

    u_gp_mean = np.cumsum(q_gp_mean, axis=0) * dz

    # Approximate displacement std from strain std.
    # This ignores covariance between spatial points, but is useful visually.
    u_gp_upper = np.cumsum(q_gp_mean + q_gp_std, axis=0) * dz
    u_gp_lower = np.cumsum(q_gp_mean - q_gp_std, axis=0) * dz

    fig, axs = plt.subplots(
        1,
        len(time_slices),
        figsize=(6 * len(time_slices), 5),
        sharey=True,
    )

    if len(time_slices) == 1:
        axs = [axs]

    fig.suptitle(
        "Average 1D String Configuration Across Seeds",
        fontsize=16,
        fontweight="bold",
    )

    for idx, step in enumerate(time_slices):
        if step >= u_true.shape[1] or step >= u_gp_mean.shape[1]:
            print(f"[WARNING] Time step {step} is out of bounds.")
            continue

        time_val = step * simulator.dt

        axs[idx].plot(
            z,
            u_true[:, step],
            "k-",
            linewidth=2.5,
            label="True Numerical PDE",
        )

        axs[idx].plot(
            z,
            u_gp_mean[:, step],
            "r--",
            linewidth=2.5,
            label=f"Mean GP-PHS Prediction ({legend_tag})",
        )

        if show_std_band:
            axs[idx].fill_between(
                z,
                u_gp_lower[:, step],
                u_gp_upper[:, step],
                alpha=0.2,
                label=r"$\pm 1\sigma$ across seeds",
            )

        axs[idx].set_title(f"Time Step = {step} ({time_val:.2f}s)", fontsize=12)
        axs[idx].set_xlabel("Spatial Coordinate (z)", fontsize=10)

        if idx == 0:
            axs[idx].set_ylabel("Displacement Profile (u)", fontsize=11)

        axs[idx].grid(True, linestyle=":", alpha=0.6)
        axs[idx].legend(loc="upper right")

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Average rollout time-slice plot saved as '{save_path}'")


def spatial_derivative_centered(f, dz, boundary_value=np.nan):
    """
    Compute df/dz using the same centered-difference structure as the simulator
    on interior points.

    Boundary values are set to NaN by default because the simulator imposes
    boundary conditions separately.
    """
    df_dz = np.full_like(f, boundary_value, dtype=float)
    df_dz[1:-1] = (f[2:] - f[:-2]) / (2.0 * dz)
    return df_dz


def _relative_l2(pred, true, eps=1e-12):
    """Relative Frobenius/L2 error."""
    return np.linalg.norm((pred - true).ravel()) / (np.linalg.norm(true.ravel()) + eps)


def _rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def reconstruct_displacement_from_strain(q, simulator):
    """
    Reconstruct displacement u from strain q = du/dz.

    This fixes u(z=0)=0. For comparison purposes this is sufficient,
    since both true and GP displacement are reconstructed the same way.
    """
    return np.cumsum(q, axis=0) * simulator.dz


def compute_rollout_error_metrics(data, p_gp, q_gp, simulator):
    """
    Computes numerical rollout errors between true PDE trajectory and GP-PHS rollout.

    Returns:
        dict of scalar metrics.
    """
    p_true = data["p"]
    q_true = data["q"]

    # Align in case solve_ivp returned fewer columns.
    T_common = min(p_true.shape[1], p_gp.shape[1], q_true.shape[1], q_gp.shape[1])
    p_true = p_true[:, :T_common]
    q_true = q_true[:, :T_common]
    p_gp = p_gp[:, :T_common]
    q_gp = q_gp[:, :T_common]

    state_true = np.concatenate([p_true, q_true], axis=0)
    state_gp = np.concatenate([p_gp, q_gp], axis=0)

    u_true = reconstruct_displacement_from_strain(q_true, simulator)
    u_gp = reconstruct_displacement_from_strain(q_gp, simulator)

    metrics = {
        "p_rel_l2": _relative_l2(p_gp, p_true),
        "q_rel_l2": _relative_l2(q_gp, q_true),
        "state_rel_l2": _relative_l2(state_gp, state_true),
        "u_rel_l2": _relative_l2(u_gp, u_true),

        "p_rmse": _rmse(p_gp, p_true),
        "q_rmse": _rmse(q_gp, q_true),
        "state_rmse": _rmse(state_gp, state_true),
        "u_rmse": _rmse(u_gp, u_true),

        "p_final_rel_l2": _relative_l2(p_gp[:, -1], p_true[:, -1]),
        "q_final_rel_l2": _relative_l2(q_gp[:, -1], q_true[:, -1]),
        "u_final_rel_l2": _relative_l2(u_gp[:, -1], u_true[:, -1]),
    }

    return metrics


def compute_energy_error_metrics(data, p_gp, q_gp, simulator):
    """
    Computes Hamiltonian trajectory error.

    Important:
        Because the system has damping and external forcing, the true energy
        is not conserved. Therefore this compares H evaluated on the GP
        rollout to H evaluated on the true rollout.
    """
    p_true = data["p"]
    q_true = data["q"]

    T_common = min(p_true.shape[1], p_gp.shape[1], q_true.shape[1], q_gp.shape[1])
    p_true = p_true[:, :T_common]
    q_true = q_true[:, :T_common]
    p_gp = p_gp[:, :T_common]
    q_gp = q_gp[:, :T_common]

    H_true = simulator.total_energy(p_true, q_true)
    H_gp = simulator.total_energy(p_gp, q_gp)

    metrics = {
        "energy_rel_l2": _relative_l2(H_gp, H_true),
        "energy_rmse": _rmse(H_gp, H_true),
        "energy_mae": _mae(H_gp, H_true),
        "energy_final_abs_error": float(abs(H_gp[-1] - H_true[-1])),
        "energy_final_rel_error": float(abs(H_gp[-1] - H_true[-1]) / (abs(H_true[-1]) + 1e-12)),
    }

    return metrics, H_true, H_gp


def make_flat_dataset_from_sim_data(data):
    """
    Reconstructs X = [p, q, z], y_p = dH/dp, y_q = dH/dq from simulator data.

    Returns:
        X_all: shape [N*T, 3]
        y_p_all: shape [N*T]
        y_q_all: shape [N*T]
    """
    p_flat = data["p"].flatten()
    q_flat = data["q"].flatten()
    y_p_all = data["dH_dp"].flatten()
    y_q_all = data["dH_dq"].flatten()

    N_pts, T_steps = data["q"].shape
    z_tiled = np.tile(data["z"].reshape(N_pts, 1), (1, T_steps))
    z_flat = z_tiled.flatten()

    X_all = np.stack([p_flat, q_flat, z_flat], axis=1)

    return X_all, y_p_all, y_q_all


def _model_mean_physical(model, X_raw_tensor):
    """
    Predicts physical-scale GP posterior mean.

    This function keeps gradients enabled so it can be used for curl/integrability.
    """
    X_mean = model.X_mean.to(X_raw_tensor.device)
    X_std = model.X_std.to(X_raw_tensor.device)
    y_mean = model.y_mean.to(X_raw_tensor.device)
    y_std = model.y_std.to(X_raw_tensor.device)

    X_norm = (X_raw_tensor - X_mean) / (X_std + 1e-6)
    mean_norm = model(X_norm).mean
    mean_phys = mean_norm * y_std + y_mean

    return mean_phys, X_norm


def compute_curl_integrability_error(
    dEdp_model,
    dEdq_model,
    X_raw,
    max_points=2048,
    batch_size=256,
    seed=0,
):
    """
    Computes the Hamiltonian integrability/curl error:

        curl = d/dq (dH/dp) - d/dp (dH/dq)

    For a true scalar Hamiltonian H(p,q,z), this should be zero.

    Since the GPs are trained separately, this metric checks whether their
    learned gradients are compatible with a shared Hamiltonian.

    Args:
        dEdp_model: GP model for dH/dp.
        dEdq_model: GP model for dH/dq.
        X_raw: numpy array [M, 3] with columns [p, q, z].
    """
    dEdp_model.eval()
    dEdq_model.eval()

    rng = np.random.default_rng(seed)

    if X_raw.shape[0] > max_points:
        idx = rng.choice(X_raw.shape[0], size=max_points, replace=False)
        X_eval = X_raw[idx]
    else:
        X_eval = X_raw

    curls = []

    for start in range(0, X_eval.shape[0], batch_size):
        end = min(start + batch_size, X_eval.shape[0])

        X_batch = torch.tensor(
            X_eval[start:end],
            dtype=torch.float64,
        )

        # ----- d/dq (dH/dp) -----
        X_p = X_batch.clone().detach().requires_grad_(True)
        mean_p, X_p_norm = _model_mean_physical(dEdp_model, X_p)
        grad_p_raw = torch.autograd.grad(
            mean_p.sum(),
            X_p,
            create_graph=False,
            retain_graph=False,
        )[0]
        d_dq_dHdp = grad_p_raw[:, 1]

        # ----- d/dp (dH/dq) -----
        X_q = X_batch.clone().detach().requires_grad_(True)
        mean_q, X_q_norm = _model_mean_physical(dEdq_model, X_q)
        grad_q_raw = torch.autograd.grad(
            mean_q.sum(),
            X_q,
            create_graph=False,
            retain_graph=False,
        )[0]
        d_dp_dHdq = grad_q_raw[:, 0]

        curl_batch = d_dq_dHdp - d_dp_dHdq
        curls.append(curl_batch.detach().cpu().numpy())

    curl = np.concatenate(curls)

    metrics = {
        "curl_mean_abs": float(np.mean(np.abs(curl))),
        "curl_rms": float(np.sqrt(np.mean(curl**2))),
        "curl_max_abs": float(np.max(np.abs(curl))),
    }

    return metrics


def gp_predict_with_uncertainty(
    model,
    likelihood,
    X_raw,
    batch_size=2048,
    include_likelihood_noise=True,
):
    """
    Predicts mean and standard deviation in physical units.

    If include_likelihood_noise=True, reports predictive uncertainty for noisy
    observations. If False, reports latent-function uncertainty.
    """
    model.eval()
    likelihood.eval()

    means = []
    stds = []

    for start in range(0, X_raw.shape[0], batch_size):
        end = min(start + batch_size, X_raw.shape[0])

        X_batch = torch.tensor(X_raw[start:end], dtype=torch.float64)
        X_norm = (X_batch - model.X_mean) / (model.X_std + 1e-6)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_dist = model(X_norm)
            pred_dist = likelihood(latent_dist) if include_likelihood_noise else latent_dist

            mean_norm = pred_dist.mean
            var_norm = pred_dist.variance.clamp_min(1e-12)

        mean_phys = mean_norm * model.y_std + model.y_mean
        std_phys = torch.sqrt(var_norm) * torch.abs(model.y_std)

        means.append(mean_phys.cpu().numpy())
        stds.append(std_phys.cpu().numpy())

    return np.concatenate(means), np.concatenate(stds)


def compute_single_calibration_metrics(y_true, mean, std):
    """
    Calibration metrics for one scalar GP output.
    """
    std = np.maximum(std, 1e-12)
    residual = y_true - mean
    z_score = residual / std

    nll = 0.5 * np.mean(
        z_score**2 + 2.0 * np.log(std) + np.log(2.0 * np.pi)
    )

    return {
        "rmse": _rmse(mean, y_true),
        "mae": _mae(mean, y_true),
        "avg_std": float(np.mean(std)),
        "coverage_68": float(np.mean(np.abs(z_score) <= 1.0)),
        "coverage_95": float(np.mean(np.abs(z_score) <= 1.96)),
        "mean_abs_z": float(np.mean(np.abs(z_score))),
        "nlpd": float(nll),
    }


def compute_dHdq_force_metrics_from_predictions(
    dHdq_pred_flat,
    dHdq_true_flat,
    simulator,
    grid_shape,
    edge=5,
):
    """
    Compute RMSE of d/dz(dH/dq) using already-computed dH/dq predictions.

    Args:
        dHdq_pred_flat:
            Flattened GP-predicted dH/dq values, shape [N_z * N_t].
        dHdq_true_flat:
            Flattened true dH/dq values, shape [N_z * N_t].
        simulator:
            Simulator object with N and dz.
        grid_shape:
            Tuple (N_z, N_t). Must match the flattening order used in
            make_flat_dataset_from_sim_data().
        edge:
            Number of boundary cells to exclude.

    Returns:
        Dictionary of force-field error metrics.
    """

    N_z, N_t = grid_shape

    assert N_z == simulator.N, "grid_shape[0] must match simulator.N"
    assert dHdq_pred_flat.shape[0] == N_z * N_t
    assert dHdq_true_flat.shape[0] == N_z * N_t

    dHdq_pred = np.asarray(dHdq_pred_flat).reshape(N_z, N_t)
    dHdq_true = np.asarray(dHdq_true_flat).reshape(N_z, N_t)

    force_pred = np.full_like(dHdq_pred, np.nan, dtype=float)
    force_true = np.full_like(dHdq_true, np.nan, dtype=float)

    for j in range(N_t):
        force_pred[:, j] = spatial_derivative_centered(
            dHdq_pred[:, j],
            simulator.dz,
            boundary_value=np.nan,
        )

        force_true[:, j] = spatial_derivative_centered(
            dHdq_true[:, j],
            simulator.dz,
            boundary_value=np.nan,
        )

    interior = slice(edge, N_z - edge)

    force_pred_int = force_pred[interior, :]
    force_true_int = force_true[interior, :]

    force_err = force_pred_int - force_true_int
    mask = np.isfinite(force_err)

    force_rmse = _rmse(
        force_pred_int[mask],
        force_true_int[mask],
    )

    force_true_rms = float(np.sqrt(np.nanmean(force_true_int**2)) + 1e-12)

    return {
        "dHdq_force_rmse": force_rmse,
        "dHdq_force_relative_rmse": force_rmse / force_true_rms,
        "dHdq_force_true_rms_scale": force_true_rms,
        "dHdq_force_max_abs_error": float(np.nanmax(np.abs(force_err))),
    }, {"dHdq_force_pred": force_pred,
        "dHdq_force_true": force_true,
        "dHdq_force_error": force_pred - force_true,}


def compute_uncertainty_calibration_metrics(
    dEdp_model,
    dEdp_likelihood,
    dEdq_model,
    dEdq_likelihood,
    X_eval,
    dHdp_true,
    dHdq_true,
    downscale_factor,
    simulator,
    grid_shape,
    seed=42,
    sampling="random",
    include_likelihood_noise=True,
):
    """
    Computes uncertainty calibration for dH/dp and dH/dq predictions
    on held-out derivative targets.
    """
    dHdp_mean, dHdp_std = gp_predict_with_uncertainty(
        dEdp_model,
        dEdp_likelihood,
        X_eval,
        include_likelihood_noise=include_likelihood_noise,
    )

    dHdq_mean, dHdq_std = gp_predict_with_uncertainty(
        dEdq_model,
        dEdq_likelihood,
        X_eval,
        include_likelihood_noise=include_likelihood_noise,
    )

    # Evaluate calibration on points not used in training.
    if sampling == "random":
        test_idx = random_z_indices(X_eval, train_fraction=downscale_factor, seed=seed)
    elif sampling == "stratified":
        test_idx = stratified_z_indices(
            X_eval,
            train_fraction=downscale_factor,
            seed=seed,
            left_region=(0.0, 2.0),
            middle_region=(2.0, 8.0),
            right_region=(8.0, 10.0),
            weights=(0.30, 0.40, 0.30),
            include_domain_boundaries=True,
        )
    else:
        raise ValueError(f"sampling {sampling} is not supported")
    heldout_mask = np.ones(X_eval.shape[0], dtype=bool)
    heldout_mask[test_idx] = False

    metrics = {
        "dHdp": compute_single_calibration_metrics(
            dHdp_true[heldout_mask],
            dHdp_mean[heldout_mask],
            dHdp_std[heldout_mask],
        ),
        "dHdq": compute_single_calibration_metrics(
            dHdq_true[heldout_mask],
            dHdq_mean[heldout_mask],
            dHdq_std[heldout_mask],
        ),
    }

    if simulator is None or grid_shape is None:
        raise ValueError(
            "compute_force_metrics=True requires simulator and grid_shape=(N_z, N_t)."
        )

    force_metrics, dHdq_summary = compute_dHdq_force_metrics_from_predictions(
        dHdq_pred_flat=dHdq_mean,
        dHdq_true_flat=dHdq_true,
        simulator=simulator,
        grid_shape=grid_shape,
    )

    # Store scalar force metrics only inside calibration.
    metrics["dHdq_force"] = force_metrics
    dHdq_summary["dHdq_mean"] = dHdq_mean
    dHdq_summary["dHdq_true"] = dHdq_true

    return metrics, dHdq_summary


def evaluate_gp_phs_model(
    data,
    simulator,
    p_gp,
    q_gp,
    dEdp_model,
    dEdq_model,
    dEdp_likelihood=None,
    dEdq_likelihood=None,
    downscale_factor=0.1,
    grid_shape=None,
    seed=42,
    sampling="random"
):
    """
    Full evaluation suite for paper reporting.

    Returns:
        metrics: nested dictionary
    """
    rollout_metrics = compute_rollout_error_metrics(data, p_gp, q_gp, simulator)
    energy_metrics, H_true, H_gp = compute_energy_error_metrics(data, p_gp, q_gp, simulator)

    X_all, y_p_all, y_q_all = make_flat_dataset_from_sim_data(data)

    curl_metrics = compute_curl_integrability_error(
        dEdp_model,
        dEdq_model,
        X_all,
    )

    metrics = {
        "rollout": rollout_metrics,
        "energy": energy_metrics,
        "integrability": curl_metrics,
    }

    if dEdp_likelihood is not None and dEdq_likelihood is not None:
        calibration_metrics, dHdq_summary = compute_uncertainty_calibration_metrics(
            dEdp_model,
            dEdp_likelihood,
            dEdq_model,
            dEdq_likelihood,
            X_all,
            y_p_all,
            y_q_all,
            downscale_factor=downscale_factor,
            simulator=simulator,
            grid_shape=grid_shape,
            seed=seed,
            sampling=sampling)

        metrics["calibration"] = calibration_metrics

    return metrics, dHdq_summary, H_true, H_gp


def print_metrics_report(metrics):
    """
    Nicely prints evaluation metrics.
    """
    print("\n================ GP-PHS Evaluation Metrics ================\n")

    print("[Rollout Error]")
    for key, value in metrics["rollout"].items():
        print(f"  {key:24s}: {value:.6e}")

    print("\n[Energy Error]")
    for key, value in metrics["energy"].items():
        print(f"  {key:24s}: {value:.6e}")

    print("\n[Curl / Integrability Error]")
    for key, value in metrics["integrability"].items():
        print(f"  {key:24s}: {value:.6e}")

    if "calibration" in metrics:
        print("\n[Uncertainty Calibration: dH/dp]")
        for key, value in metrics["calibration"]["dHdp"].items():
            print(f"  {key:24s}: {value:.6e}")

        print("\n[Uncertainty Calibration: dH/dq]")
        for key, value in metrics["calibration"]["dHdq"].items():
            print(f"  {key:24s}: {value:.6e}")

        if "dHdq_force" in metrics["calibration"]:
            print("\n[Force-Field Error: d/dz(dH/dq)]")
            for key, value in metrics["calibration"]["dHdq_force"].items():
                print(f"  {key:32s}: {value:.6e}")

    print("\n============================================================\n")
