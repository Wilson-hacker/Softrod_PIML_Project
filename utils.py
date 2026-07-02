import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
import gpytorch


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


def compute_uncertainty_calibration_metrics(
    dEdp_model,
    dEdp_likelihood,
    dEdq_model,
    dEdq_likelihood,
    X_eval,
    y_p_eval,
    y_q_eval,
    max_points=5000,
    seed=1,
    include_likelihood_noise=True,
):
    """
    Computes uncertainty calibration for dH/dp and dH/dq predictions
    on held-out derivative targets.
    """
    rng = np.random.default_rng(seed)

    if X_eval.shape[0] > max_points:
        idx = rng.choice(X_eval.shape[0], size=max_points, replace=False)
        X_eval = X_eval[idx]
        y_p_eval = y_p_eval[idx]
        y_q_eval = y_q_eval[idx]

    p_mean, p_std = gp_predict_with_uncertainty(
        dEdp_model,
        dEdp_likelihood,
        X_eval,
        include_likelihood_noise=include_likelihood_noise,
    )

    q_mean, q_std = gp_predict_with_uncertainty(
        dEdq_model,
        dEdq_likelihood,
        X_eval,
        include_likelihood_noise=include_likelihood_noise,
    )

    return {
        "dHdp": compute_single_calibration_metrics(y_p_eval, p_mean, p_std),
        "dHdq": compute_single_calibration_metrics(y_q_eval, q_mean, q_std),
    }


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
        # Evaluate calibration on points not used in training.
        num_test = int(X_all.shape[0] * downscale_factor)
        rng = np.random.default_rng(seed=42)
        test_idx = rng.choice(X_all.shape[0], size=num_test, replace=False)

        heldout_mask = np.ones(X_all.shape[0], dtype=bool)
        heldout_mask[test_idx] = False

        X_eval = X_all[heldout_mask]
        y_p_eval = y_p_all[heldout_mask]
        y_q_eval = y_q_all[heldout_mask]

        calibration_metrics = compute_uncertainty_calibration_metrics(
            dEdp_model,
            dEdp_likelihood,
            dEdq_model,
            dEdq_likelihood,
            X_eval,
            y_p_eval,
            y_q_eval,
        )

        metrics["calibration"] = calibration_metrics

    return metrics, H_true, H_gp


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

    print("\n============================================================\n")
