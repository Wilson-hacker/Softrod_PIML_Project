import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator import SoftRodSimulator
from model import train_gp_model
from utils import (
    stratified_z_indices,
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

def get_file_tag(kernel_type, param_dict):
    if kernel_type == "baseline":
        return "baseline"
    elif kernel_type == "SD":
        return f"{param_dict["strain_kernel"]}_{param_dict["spatial_kernel"]}"
    elif kernel_type == "gibbs":
        return "warped"
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def get_caption_tag(kernel_type, param_dict):
    if kernel_type == "baseline":
        return "baseline"
    elif kernel_type == "SD":
        return f"SD [{param_dict["strain_kernel"]}+{param_dict["spatial_kernel"]}]"
    elif kernel_type == "gibbs":
        return "warped"
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def run_pipeline(param_dict, seed=42, kernel_type="baseline"):
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
        seed=seed,
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

    print("\n--- Step 3: Running Dynamic Rollout and Time-Slice Comparisons ---")
    file_tag = get_file_tag(kernel_type, param_dict)
    title_tag = get_caption_tag(kernel_type, param_dict)
    p_gp, q_gp = simulate_gp_phs(sim, dEdp_model, dEdq_model, get_test_q_config)
    test_data = sim.generate_dataset(get_test_q_config)
    plot_time_slices(test_data, q_gp, sim, time_slices=[0, 50, 100], save_path=f"./plots/string_slices_{file_tag}.png", legend_tag=title_tag)

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
        grid_shape=(N_pts, T_steps),
        seed=seed
    )
    print_metrics_report(metrics)

    print("\n--- Step 5: Doing Error Analysis on Effort, Force and Strain ---")
    _ = plot_effort_force_rmse_and_qmax_over_time(
        dHdq_summary=dHdq_summary,
        q_gp=q_gp,
        simulator=sim,
        data=test_data,
        edge=5,
        title=f"Effort / Force Error and Strain Growth ({title_tag})",
        save_path=f"./plots/force_err_vs_t_{file_tag}.png",
    )
    _ = plot_force_field_error_heatmap(
        dHdq_summary=dHdq_summary,
        simulator=sim,
        data=test_data,
        grid_shape=(N_pts, T_steps),
        edge=5,
        log_scale=False,
        title=fr"Force-Field Error: $|\Delta \partial_z(\partial_q H)|$ ({title_tag})",
        save_path=f"./plots/force_field_err_{file_tag}.png",
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
        "spatial_kernel": "rbf",
        "matern_nu": 2.5,
        "changepoint_location": 5.0,
        "changepoint_width": 0.5,
        "n_freqs": 10
    }
    run_pipeline(param_dict, kernel_type="SD")
    # run_diagnostic(param_dict)