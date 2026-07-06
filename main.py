import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator import SoftRodSimulator
from model import train_gp_model
from utils import (
    stratified_z_indices,
    random_z_indices,
    simulate_gp_phs,
    plot_comparison,
    plot_time_slices,
    plot_average_time_slices,
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
    if kernel_type == "SD":
        return f"{param_dict["strain_kernel"]}_{param_dict["spatial_kernel"]}"
    elif kernel_type == "baseline" or kernel_type == "warped":
        return kernel_type
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def get_caption_tag(kernel_type, param_dict):
    if kernel_type == "SD":
        return f"SD [{param_dict["strain_kernel"]}+{param_dict["spatial_kernel"]}]"
    elif kernel_type == "baseline" or kernel_type == "warped":
        return kernel_type
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def extract_selected_metrics(metrics):
    """
    Extract only the metrics requested for multi-seed averaging.
    """
    return {
        "q_rel_l2": metrics["rollout"]["q_rel_l2"],
        "u_rel_l2": metrics["rollout"]["u_rel_l2"],
        "energy_rel_l2": metrics["energy"]["energy_rel_l2"],
        "curl_rms": metrics["integrability"]["curl_rms"],
        "dHdq_rmse": metrics["calibration"]["dHdq"]["rmse"],
        "dHdq_force_rmse": metrics["calibration"]["dHdq_force"]["dHdq_force_rmse"],
    }


def run_pipeline(
    param_dict,
    seed=42,
    kernel_type="baseline",
    sampling="random",
    train_sample_fraction=0.06,
    make_plots=True,
    return_outputs=True,
    verbose=False,
):
    print("\n--- Step 1: Solving Soft Robot PDE via Numerical Integrators ---")
    sim = SoftRodSimulator(
        L=10.0,
        N=60,
        T=5.5,
        dt=0.05,
        num_hinges=param_dict["num_hinges"],
        stiffness_profile=param_dict["stiffness_profile"],
    )

    data = sim.generate_dataset(get_train_q_config)

    p_flat = data["p"].flatten()
    q_flat = data["q"].flatten()
    dH_dq_target = data["dH_dq"].flatten()
    dH_dp_target = data["dH_dp"].flatten()

    N_pts, T_steps = data["q"].shape
    z_tiled = np.tile(data["z"].reshape(N_pts, 1), (1, T_steps))
    z_flat = z_tiled.flatten()

    X = np.stack([p_flat, q_flat, z_flat], axis=1)

    if sampling == "random":
        train_idx = random_z_indices(X, train_fraction=train_sample_fraction, seed=seed)
    elif sampling == "stratified":
        train_idx = stratified_z_indices(
            X,
            train_fraction=train_sample_fraction,
            seed=seed,
            left_region=(0.0, 2.0),
            middle_region=(2.0, 8.0),
            right_region=(8.0, 10.0),
            weights=(0.30, 0.40, 0.30),
            include_domain_boundaries=True,
        )
    else:
        raise ValueError(f"sampling {sampling} is not supported")

    X_train = torch.tensor(X[train_idx], dtype=torch.float64)
    y_dEdp_train = torch.tensor(dH_dp_target[train_idx], dtype=torch.float64)
    y_dEdq_train = torch.tensor(dH_dq_target[train_idx], dtype=torch.float64)

    print(f"Data Generation Complete. GP Input Shape: {X_train.shape}")

    # Z-score normalization
    X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-6)

    y_p_mean, y_p_std = y_dEdp_train.mean(), y_dEdp_train.std()
    y_p_train_norm = (y_dEdp_train - y_p_mean) / (y_p_std + 1e-6)

    y_q_mean, y_q_std = y_dEdq_train.mean(), y_dEdq_train.std()
    y_q_train_norm = (y_dEdq_train - y_q_mean) / (y_q_std + 1e-6)

    print("\n--- Step 2: Training GP-PHS Component Models ---")

    dEdp_model, dEdp_lh = train_gp_model(
        X_train_norm,
        y_p_train_norm,
        param_dict,
        kernel_type="baseline",
        training_iter=1000,
        patience=15,
        verbose=verbose,
    )

    dEdq_model, dEdq_lh = train_gp_model(
        X_train_norm,
        y_q_train_norm,
        param_dict,
        kernel_type=kernel_type,
        training_iter=1000,
        patience=15,
        verbose=verbose,
    )

    # Store normalization stats
    dEdp_model.X_mean, dEdp_model.X_std = X_mean, X_std
    dEdp_model.y_mean, dEdp_model.y_std = y_p_mean, y_p_std

    dEdq_model.X_mean, dEdq_model.X_std = X_mean, X_std
    dEdq_model.y_mean, dEdq_model.y_std = y_q_mean, y_q_std

    print("\n--- Step 3: Running Dynamic Rollout and Time-Slice Comparisons ---")

    file_tag = get_file_tag(kernel_type, param_dict)
    title_tag = get_caption_tag(kernel_type, param_dict)

    p_gp, q_gp = simulate_gp_phs(
        sim,
        dEdp_model,
        dEdq_model,
        get_test_q_config,
    )

    test_data = sim.generate_dataset(get_test_q_config)

    if make_plots:
        plot_time_slices(
            test_data,
            q_gp,
            sim,
            time_slices=[0, 50, 100],
            save_path=f"./plots/string_slices_{file_tag}_seed{seed}.png",
            legend_tag=f"{title_tag}, seed={seed}",
        )

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
        downscale_factor=train_sample_fraction,
        grid_shape=(N_pts, T_steps),
        seed=seed,
        sampling=sampling
    )

    if verbose:
        print_metrics_report(metrics)

    if make_plots:
        print("\n--- Step 5: Doing Error Analysis on Effort, Force and Strain ---")

        _ = plot_effort_force_rmse_and_qmax_over_time(
            dHdq_summary=dHdq_summary,
            q_gp=q_gp,
            simulator=sim,
            data=test_data,
            edge=5,
            title=f"Effort / Force Error and Strain Growth ({title_tag}, seed={seed})",
            save_path=f"./plots/force_err_vs_t_{file_tag}_seed{seed}.png",
        )

        _ = plot_force_field_error_heatmap(
            dHdq_summary=dHdq_summary,
            simulator=sim,
            data=test_data,
            grid_shape=(N_pts, T_steps),
            edge=5,
            log_scale=False,
            title=fr"Force-Field Error: $|\Delta \partial_z(\partial_q H)|$ ({title_tag}, seed={seed})",
            save_path=f"./plots/force_field_err_{file_tag}_seed{seed}.png",
        )

    print("Pipeline execution completed successfully.\n")

    if return_outputs:
        return {
            "seed": seed,
            "metrics": metrics,
            "dHdq_summary": dHdq_summary,
            "H_true": H_true,
            "H_gp": H_gp,
            "p_gp": p_gp,
            "q_gp": q_gp,
            "test_data": test_data,
            "simulator": sim,
            "file_tag": file_tag,
            "title_tag": title_tag,
        }
    else:
        return None


def run_pipeline_multiple_seeds(
    param_dict,
    seeds=(0, 1, 2, 3, 4),
    kernel_type="baseline",
    train_sample_fraction=0.06,
    sampling="random",
    make_individual_plots=False,
    plot_average_rollout=True,
    time_slices=(0, 50, 100),
    save_average_plot=True,
    show_std_band=False,
):
    """
    Runs run_pipeline() for multiple random seeds, averages selected metrics,
    and optionally plots the average predicted rollout trajectory.

    Returns:
        summary:
            Dictionary containing average metrics, std metrics, per-seed metrics,
            q_gp mean/std, and raw seed outputs.
    """

    all_outputs = []
    per_seed_metrics = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Running seed {seed}")
        print("=" * 80)

        out = run_pipeline(
            param_dict=param_dict,
            train_sample_fraction=train_sample_fraction,
            seed=seed,
            kernel_type=kernel_type,
            make_plots=make_individual_plots,
            return_outputs=True,
            sampling=sampling
        )

        all_outputs.append(out)

        selected = extract_selected_metrics(out["metrics"])
        selected["seed"] = seed
        per_seed_metrics.append(selected)

    # ------------------------------------------------------------
    # Average selected metrics
    # ------------------------------------------------------------
    metric_keys = [
        "q_rel_l2",
        "u_rel_l2",
        "energy_rel_l2",
        "curl_rms",
        "dHdq_rmse",
        "dHdq_force_rmse",
    ]

    avg_metrics = {}
    std_metrics = {}

    for key in metric_keys:
        values = np.array([m[key] for m in per_seed_metrics], dtype=float)
        avg_metrics[key] = float(np.mean(values))
        std_metrics[key] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # ------------------------------------------------------------
    # Average predicted rollout q over seeds
    # ------------------------------------------------------------
    q_stack = np.stack([out["q_gp"] for out in all_outputs], axis=0)
    p_stack = np.stack([out["p_gp"] for out in all_outputs], axis=0)

    q_gp_mean = np.mean(q_stack, axis=0)
    q_gp_std = np.std(q_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(q_gp_mean)

    p_gp_mean = np.mean(p_stack, axis=0)
    p_gp_std = np.std(p_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(p_gp_mean)

    # Use first run's sim/test_data, since deterministic simulator and test IC
    # should be identical across seeds.
    ref_output = all_outputs[0]
    sim = ref_output["simulator"]
    test_data = ref_output["test_data"]
    title_tag = ref_output["title_tag"]
    file_tag = ref_output["file_tag"]

    if plot_average_rollout:
        if save_average_plot:
            save_path = f"./plots/average_string_slices_{file_tag}_nseeds{len(seeds)}.png"
        else:
            save_path = None

        if save_path is None:
            # Same plotting logic, but display instead of saving.
            plot_average_time_slices(
                data=test_data,
                q_gp_mean=q_gp_mean,
                q_gp_std=q_gp_std,
                simulator=sim,
                time_slices=list(time_slices),
                legend_tag=f"{title_tag}, n={len(seeds)} seeds",
                show_std_band=show_std_band,
            )
        else:
            plot_average_time_slices(
                data=test_data,
                q_gp_mean=q_gp_mean,
                q_gp_std=q_gp_std,
                simulator=sim,
                time_slices=list(time_slices),
                save_path=save_path,
                legend_tag=f"{title_tag}, n={len(seeds)} seeds",
                show_std_band=show_std_band,
            )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Multi-seed average metrics")
    print("=" * 80)

    for key in metric_keys:
        print(f"{key:24s}: {avg_metrics[key]:.6e} ± {std_metrics[key]:.6e}")

    print("=" * 80 + "\n")

    summary = {
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "per_seed_metrics": per_seed_metrics,
        "q_gp_mean": q_gp_mean,
        "q_gp_std": q_gp_std,
        "p_gp_mean": p_gp_mean,
        "p_gp_std": p_gp_std,
        "all_outputs": all_outputs,
    }

    return summary


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
    run_pipeline_multiple_seeds(param_dict, seeds=(6, 42, 79, 123, 456), kernel_type="baseline",
                                sampling="stratified", train_sample_fraction=0.06)
    # run_pipeline(param_dict, kernel_type="baseline", seed=42, verbose=True, make_plots=True)
    # run_diagnostic(param_dict)