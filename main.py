import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator import SoftRodSimulator
from model import train_gp_model
from utils import (
    simulate_gp_phs,
    plot_comparison,
    plot_time_slices,
    evaluate_gp_phs_model,
    print_metrics_report,
    plot_effort_force_rmse_and_qmax_over_time,
    plot_force_field_error_heatmap
)
from diagnostic import (plot_q_dHdq_force_diagnostic,
                        plot_boundary_vs_interior_q,
                        scan_rollout_rhs_error,
                        inspect_rhs_at_step)


def get_train_q_config(z_arr, length):
    return -4.0 * (z_arr - length / 2) * np.exp(-(z_arr - length / 2) ** 2)

def get_test_q_config(z_arr, length):
    return get_train_q_config(z_arr, length)
    # return -3.0 * (z_arr - length * 0.4) * np.exp(-0.8 * (z_arr - length * 0.4) ** 2)

def stratified_z_indices(
    X,
    train_fraction=0.1,
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

def run_pipeline(tag_str, param_dict, kernel_type="baseline"):
    print("\n--- Step 1: Solving Soft Robot PDE via Numerical Integrators ---")
    sim = SoftRodSimulator(L=10.0, N=60, T=5.5, dt=0.05, num_hinges=param_dict["num_hinges"], stiffness_profile=param_dict["stiffness_profile"])
    data = sim.generate_dataset(get_train_q_config)

    p_flat = data['p'].flatten()
    q_flat = data['q'].flatten()
    dH_dq_target = data['dH_dq'].flatten()
    dH_dp_target = data['dH_dp'].flatten()

    # Tile the spatial array to match the time dimension, then flatten
    N_pts, T_steps = data['q'].shape
    z_tiled = np.tile(data['z'].reshape(N_pts, 1), (1, T_steps))
    z_flat = z_tiled.flatten()

    # X matrix is now shape [N_total, 3] -> (p, q, z)
    X = np.stack([p_flat, q_flat, z_flat], axis=1)

    downscale_factor = 0.06
    train_idx = stratified_z_indices(
        X,
        train_fraction=downscale_factor,
        seed=42,
        left_region=(0.0, 2.0),
        middle_region=(2.0, 8.0),
        right_region=(8.0, 10.0),
        weights=(0.30, 0.40, 0.30),
        include_domain_boundaries=True,
    )
    # num_train = int(X.shape[0] * downscale_factor)
    # rng = np.random.default_rng(seed=42)
    # train_idx = rng.choice(X.shape[0], size=num_train, replace=False)

    X_train = torch.tensor(X[train_idx], dtype=torch.float64)
    y_dEdp_train = torch.tensor(dH_dp_target[train_idx], dtype=torch.float64)
    y_dEdq_train = torch.tensor(dH_dq_target[train_idx], dtype=torch.float64)

    print(f"Data Generation Complete. GP Input Shape: {X_train.shape}")

    # Z-Score Normalization
    X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-6)

    y_p_mean, y_p_std = y_dEdp_train.mean(), y_dEdp_train.std()
    y_p_train_norm = (y_dEdp_train - y_p_mean) / (y_p_std + 1e-6)

    y_q_mean, y_q_std = y_dEdq_train.mean(), y_dEdq_train.std()
    y_q_train_norm = (y_dEdq_train - y_q_mean) / (y_q_std + 1e-6)

    print("\n--- Step 2: Training GP-PHS Component Models ---")
    dEdp_model, dEdp_lh = train_gp_model(X_train_norm, y_p_train_norm, param_dict, kernel_type="baseline", training_iter=1000, patience=15)
    dEdq_model, dEdq_lh = train_gp_model(X_train_norm, y_q_train_norm, param_dict, kernel_type=kernel_type, training_iter=1000, patience=15)

    # Store normalization stats directly on the models for easy access in utils.py
    dEdp_model.X_mean, dEdp_model.X_std = X_mean, X_std
    dEdp_model.y_mean, dEdp_model.y_std = y_p_mean, y_p_std
    dEdq_model.X_mean, dEdq_model.X_std = X_mean, X_std
    dEdq_model.y_mean, dEdq_model.y_std = y_q_mean, y_q_std

    # print("\n--- Step 3: Projecting and Evaluating System Approximations ---")
    # plot_comparison(data, dEdp_model, dEdq_model, sim, save_path=f"./plots/H_landscape_{tag_str}.png", legend_tag=tag_str)

    print("\n--- Step 3: Running Dynamic Rollout and Time-Slice Comparisons ---")
    p_gp, q_gp = simulate_gp_phs(sim, dEdp_model, dEdq_model, get_test_q_config)
    test_data = sim.generate_dataset(get_test_q_config)
    plot_time_slices(test_data, q_gp, sim, time_slices=[0, 50, 100], save_path=f"./plots/string_slices_{tag_str}.png", legend_tag=tag_str)

    print("\n--- Step 4: Computing Paper-Ready Evaluation Metrics ---")
    metrics, dHdq_summary, H_true, H_gp = evaluate_gp_phs_model(
        data=test_data,
        simulator=sim,
        p_gp=p_gp,
        q_gp=q_gp,
        dEdp_model=dEdp_model,
        dEdq_model=dEdq_model,
        dEdp_likelihood=dEdp_lh,
        dEdq_likelihood=dEdq_lh,
        downscale_factor=downscale_factor,
        grid_shape=(N_pts, T_steps)
    )
    print_metrics_report(metrics)

    _ = plot_effort_force_rmse_and_qmax_over_time(
        dHdq_summary=dHdq_summary,
        q_gp=q_gp,
        simulator=sim,
        data=test_data,
        edge=5,
        title="Effort / Force Error and Strain Growth",
        save_path=None,
    )
    _ = plot_force_field_error_heatmap(
        dHdq_summary=dHdq_summary,
        simulator=sim,
        data=test_data,
        grid_shape=(N_pts, T_steps),
        edge=5,
        log_scale=False,
        title=r"Force-Field Error: $|\Delta \partial_z(\partial_q H)|$",
        save_path=None,
    )

    print("Pipeline execution completed successfully.\n")


def run_diagnostic(param_dict, kernel_type="baseline"):
    sim = SoftRodSimulator(L=10.0, N=60, T=5.5, dt=0.05, num_hinges=param_dict["num_hinges"],
                           stiffness_profile=param_dict["stiffness_profile"])
    data = sim.generate_dataset(get_train_q_config)

    p_flat = data['p'].flatten()
    q_flat = data['q'].flatten()
    dH_dq_target = data['dH_dq'].flatten()
    dH_dp_target = data['dH_dp'].flatten()

    # Tile the spatial array to match the time dimension, then flatten
    N_pts, T_steps = data['q'].shape
    z_tiled = np.tile(data['z'].reshape(N_pts, 1), (1, T_steps))
    z_flat = z_tiled.flatten()

    # X matrix is now shape [N_total, 3] -> (p, q, z)
    X = np.stack([p_flat, q_flat, z_flat], axis=1)

    downscale_factor = 0.1
    num_train = int(X.shape[0] * downscale_factor)
    rng = np.random.default_rng(seed=42)
    train_idx = rng.choice(X.shape[0], size=num_train, replace=False)

    X_train = torch.tensor(X[train_idx], dtype=torch.float64)
    y_dEdp_train = torch.tensor(dH_dp_target[train_idx], dtype=torch.float64)
    y_dEdq_train = torch.tensor(dH_dq_target[train_idx], dtype=torch.float64)

    # Z-Score Normalization
    X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-6)

    y_p_mean, y_p_std = y_dEdp_train.mean(), y_dEdp_train.std()
    y_p_train_norm = (y_dEdp_train - y_p_mean) / (y_p_std + 1e-6)

    y_q_mean, y_q_std = y_dEdq_train.mean(), y_dEdq_train.std()
    y_q_train_norm = (y_dEdq_train - y_q_mean) / (y_q_std + 1e-6)

    dEdp_model, dEdp_lh = train_gp_model(X_train_norm, y_p_train_norm, param_dict, kernel_type="baseline",
                                         training_iter=1000, patience=15)
    dEdq_model, dEdq_lh = train_gp_model(X_train_norm, y_q_train_norm, param_dict, kernel_type=kernel_type,
                                         training_iter=1000, patience=15)

    # Store normalization stats directly on the models for easy access in utils.py
    dEdp_model.X_mean, dEdp_model.X_std = X_mean, X_std
    dEdp_model.y_mean, dEdp_model.y_std = y_p_mean, y_p_std
    dEdq_model.X_mean, dEdq_model.X_std = X_mean, X_std
    dEdq_model.y_mean, dEdq_model.y_std = y_q_mean, y_q_std

    p_gp, q_gp = simulate_gp_phs(sim, dEdp_model, dEdq_model, get_test_q_config)
    test_data = sim.generate_dataset(get_test_q_config)

    # stiffness_activity_diagnostic(data, sim)
    # ph_force_diagnostic(data, sim)
    # gradient_error_by_z(data, sim, dEdq_model)
    # plot_q_dHdq_force_diagnostic(test_data, p_gp, q_gp, sim, dEdq_model, time_steps=(31, 32, 33))
    plot_boundary_vs_interior_q(q_gp, test_data, sim)
    scan_rollout_rhs_error(test_data, p_gp, q_gp, sim, dEdq_model, dpdt_bad_threshold=0.6)
    for step in [30, 31, 32]:
        inspect_rhs_at_step(test_data, p_gp, q_gp, sim, dEdq_model, step)


if __name__ == '__main__':
    param_dict = {
        "stiffness_profile": "periodic",
        "num_hinges": 4,
        "strain_kernel": "rbf",
        "spatial_kernel": "periodic",
        "matern_nu": 2.5,
        "changepoint_location": 5.0,
        "changepoint_width": 0.5,
        "n_freqs": 10
    }
    run_pipeline("new_periodic", param_dict, kernel_type="SD")
    # run_diagnostic(param_dict)