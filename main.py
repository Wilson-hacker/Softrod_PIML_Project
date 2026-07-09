import os
import copy
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator import SoftRodSimulator
from model import train_gp_model
from utils import (
    stratified_z_indices,
    random_z_indices,
    normalize_changepoint_params,
    simulate_gp_phs,
    compute_effort_force_error_curves,
    compute_rollout_failure_stats,
    plot_comparison,
    plot_rollout_time_slices,
    evaluate_gp_phs_model,
    print_metrics_report,
    plot_effort_force_rmse_and_qmax_curves,
    plot_force_field_error_heatmap,
    plot_overlaid_average_rollout_time_slices
)
from diagnostic import (stiffness_activity_diagnostic,
                        plot_q_dHdq_force_diagnostic,
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
        return f"{param_dict["stiffness_profile"]}_stiff_{param_dict["strain_kernel"]}_{param_dict["spatial_kernel"]}"
    elif kernel_type == "baseline" or kernel_type == "warped":
        return f"{param_dict["stiffness_profile"]}_stiff_{kernel_type}"
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def get_caption_tag(kernel_type, param_dict):
    if kernel_type == "SD":
        spatial_kernel_str = fr"{param_dict["spatial_kernel"]} $\nu={param_dict["matern_nu"]:.1f}$" \
            if param_dict["spatial_kernel"] == "matern" else param_dict["spatial_kernel"]
        return f"SD-{spatial_kernel_str}"
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


def print_metrics_table(metrics_table):
    print(f"{'Trial num.':>10} | {'q_rel_l2':>8} | {'u_rel_l2':>8} | {'E_rel_l2':>8} | {'curl_rms':>8} | {'dHdq_rmse':>9} | {'force_rmse':>10} | {'%fail.':>8} | {'max_q_ratio':>11}")
    print("-" * 110)
    for i, metric in enumerate(metrics_table):
        print(f"{i:>10d} | "
              f"{metric[0]:>8.2e} | "
              f"{metric[1]:>8.2e} | "
              f"{metric[2]:>8.2e} | "
              f"{metric[3]:>8.2e} | "
              f"{metric[4]:>9.2e} | "
              f"{metric[5]:>10.2e} | "
              f"{metric[6]:>8.2e} | "
              f"{metric[7]:>11.2e}")


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

    param_dict_copy = copy.deepcopy(param_dict)
    if kernel_type == "SD" and param_dict["spatial_kernel"] == "changepoint":
        init_location_norm, init_width_norm = normalize_changepoint_params(
            param_dict["changepoint_location"], param_dict["changepoint_width"], X_mean, X_std)
        param_dict_copy["changepoint_location"], param_dict_copy["changepoint_width"] = init_location_norm, init_width_norm

    print("\n--- Step 2: Training GP-PHS Component Models ---")

    dEdp_model, dEdp_lh = train_gp_model(
        X_train_norm,
        y_p_train_norm,
        param_dict_copy,
        kernel_type="baseline",
        training_iter=1000,
        patience=15,
        verbose=verbose,
    )

    dEdq_model, dEdq_lh = train_gp_model(
        X_train_norm,
        y_q_train_norm,
        param_dict_copy,
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
        plot_rollout_time_slices(
            data=test_data,
            q_pred=q_gp,
            simulator=sim,
            time_slices=[0, 50, 100],
            save_path=f"./plots/string_slices_{file_tag}_seed{seed}.png",
            legend_tag=f"{title_tag}, seed={seed}",
            title=f"1D String Configuration ({param_dict["stiffness_profile"]} stiffness)",
            pred_label_prefix="GP-PHS Prediction",
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

        curves = compute_effort_force_error_curves(
            dHdq_summary=dHdq_summary,
            q_gp=q_gp,
            simulator=sim,
            data=test_data,
            edge=5,
        )

        plot_effort_force_rmse_and_qmax_curves(
            curves=curves,
            title=f"Effort / Force Error and Strain Growth ({param_dict["stiffness_profile"]} stiffness, {title_tag}, seed={seed}))",
            save_path=f"./plots/force_err(t)_{file_tag}_seed{seed}.png",
        )

        plot_force_field_error_heatmap(
            simulator=sim,
            data=test_data,
            dHdq_summary=dHdq_summary,
            grid_shape=(N_pts, T_steps),
            edge=5,
            log_scale=False,
            title=fr"Force-Field Error: $|\Delta \partial_z(\partial_q H)|$ ({param_dict["stiffness_profile"]} stiffness, {title_tag}, seed={seed})",
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
    plot_average_error_curves=True,
    plot_average_heatmap=True,
    time_slices=(0, 50, 100),
    save_average_plot=True,
    q_growth_failure_threshold=3.0,
    edge=5,
):
    """
    Runs run_pipeline() for multiple random seeds, averages selected metrics,
    plots average rollout/error diagnostics, and reports failure rate.

    Failure definition:
        A seed fails if

            max |q_GP| / max |q_true| > q_growth_failure_threshold

        using only the interior region.
    """

    all_outputs = []
    per_seed_metrics = []
    per_seed_curves = []
    force_error_abs_list = []

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
            sampling=sampling,
        )

        all_outputs.append(out)

        selected = extract_selected_metrics(out["metrics"])
        selected["seed"] = seed
        per_seed_metrics.append(selected)

        curves = compute_effort_force_error_curves(
            dHdq_summary=out["dHdq_summary"],
            q_gp=out["q_gp"],
            simulator=out["simulator"],
            data=out["test_data"],
            edge=edge,
        )

        per_seed_curves.append(curves)
        force_error_abs_list.append(curves["force_error_abs"])

    # ------------------------------------------------------------
    # Average selected scalar metrics
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
    # Average predicted rollout q and p over seeds
    # ------------------------------------------------------------
    q_stack = np.stack([out["q_gp"] for out in all_outputs], axis=0)
    p_stack = np.stack([out["p_gp"] for out in all_outputs], axis=0)

    q_gp_mean = np.mean(q_stack, axis=0)
    p_gp_mean = np.mean(p_stack, axis=0)

    # Kept only for possible diagnostics; not plotted as a band.
    q_gp_std = np.std(q_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(q_gp_mean)
    p_gp_std = np.std(p_stack, axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(p_gp_mean)

    # Reference deterministic objects
    ref_output = all_outputs[0]
    sim = ref_output["simulator"]
    test_data = ref_output["test_data"]
    title_tag = ref_output["title_tag"]
    file_tag = ref_output["file_tag"]

    # ------------------------------------------------------------
    # Average effort/force curves
    # ------------------------------------------------------------
    t_arr = per_seed_curves[0]["t"]

    dHdq_rmse_stack = np.stack([c["dHdq_rmse_t"] for c in per_seed_curves], axis=0)
    force_rmse_stack = np.stack([c["force_rmse_t"] for c in per_seed_curves], axis=0)
    qmax_gp_stack = np.stack([c["qmax_gp_t"] for c in per_seed_curves], axis=0)

    avg_curves = {
        "t": t_arr,
        "dHdq_rmse_t": np.mean(dHdq_rmse_stack, axis=0),
        "force_rmse_t": np.mean(force_rmse_stack, axis=0),
        "qmax_gp_t": np.mean(qmax_gp_stack, axis=0),
        "qmax_true_t": per_seed_curves[0]["qmax_true_t"],
    }

    # ------------------------------------------------------------
    # Average force-field error heatmap
    # ------------------------------------------------------------
    force_error_abs_stack = np.stack(force_error_abs_list, axis=0)
    force_error_abs_mean = np.mean(force_error_abs_stack, axis=0)

    # Optional but useful to keep
    force_error_abs_max = np.max(force_error_abs_stack, axis=0)

    # ------------------------------------------------------------
    # Failure statistics
    # ------------------------------------------------------------
    failure_stats = compute_rollout_failure_stats(
        all_outputs=all_outputs,
        edge=edge,
        q_growth_failure_threshold=q_growth_failure_threshold,
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    if plot_average_rollout:
        save_path = (
            f"./plots/avg_string_slices_{file_tag}.png"
            if save_average_plot
            else None
        )

        plot_rollout_time_slices(
            data=test_data,
            q_pred=q_gp_mean,
            simulator=sim,
            time_slices=list(time_slices),
            save_path=save_path,
            legend_tag=f"{title_tag}, {len(seeds)} seeds",
            title=f"Average 1D String Configuration Across Seeds ({param_dict["stiffness_profile"]} stiffness)",
            pred_label_prefix="Mean GP-PHS Prediction",
        )

    if plot_average_error_curves:
        save_path = (
            f"./plots/avg_force_err(t)_{file_tag}.png"
            if save_average_plot
            else None
        )

        plot_effort_force_rmse_and_qmax_curves(
            curves=avg_curves,
            title=f"Average Effort / Force Error and Strain Growth ({param_dict["stiffness_profile"]} stiffness, {title_tag})",
            save_path=save_path,
        )

    if plot_average_heatmap:
        save_path = (
            f"./plots/avg_force_field_err_{file_tag}.png"
            if save_average_plot
            else None
        )

        plot_force_field_error_heatmap(
            simulator=sim,
            data=test_data,
            force_error_abs=force_error_abs_mean,
            edge=edge,
            log_scale=False,
            title=fr"Mean Force-Field Error: $|\Delta \partial_z(\partial_q H)|$ ({param_dict["stiffness_profile"]} stiffness, {title_tag})",
            save_path=save_path,
        )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Multi-seed average metrics")
    print("=" * 80)

    for key in metric_keys:
        print(f"{key:24s}: {avg_metrics[key]:.6e} ± {std_metrics[key]:.6e}")

    print("\nRollout failure statistics")
    print("-" * 80)
    print(
        "Failure definition: "
        f"max|q_GP| / max|q_true| > {q_growth_failure_threshold:.2f}"
    )
    print(
        f"Failures: {failure_stats['num_failures']} / {failure_stats['num_seeds']} "
        f"= {100.0 * failure_stats['failure_rate']:.1f}%"
    )
    print(f"Failed seeds: {failure_stats['failed_seeds']}")
    print(
        "q-growth ratio: "
        f"mean = {failure_stats['q_growth_ratio_mean']:.3f}, "
        f"std = {failure_stats['q_growth_ratio_std']:.3f}, "
        f"max = {failure_stats['q_growth_ratio_max']:.3f}"
    )

    print("=" * 80 + "\n")

    summary = {
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "per_seed_metrics": per_seed_metrics,

        "failure_stats": failure_stats,

        "q_gp_mean": q_gp_mean,
        "q_gp_std": q_gp_std,
        "p_gp_mean": p_gp_mean,
        "p_gp_std": p_gp_std,

        "avg_curves": avg_curves,
        "per_seed_curves": per_seed_curves,

        "force_error_abs_mean": force_error_abs_mean,
        "force_error_abs_max": force_error_abs_max,

        "all_outputs": all_outputs,
    }

    return summary


def run_model_configurations_with_overlay(
    param_dict,
    test_config,
    seeds=(6, 42, 79, 123, 456),
    sampling="stratified",
    train_sample_fraction=0.06,
    time_slices=(0, 50, 100),
    q_growth_failure_threshold=3.0,
    edge=5,
    save_overlay_path=None,
    plot_overlay=True,
):
    """
    Runs run_pipeline_multiple_seeds() across several model configurations,
    collects averaged metrics, and plots overlaid averaged rollout trajectories.

    Args:
        param_dict:
            Base parameter dictionary. This function copies it for each config.
        test_config:
            Iterable of tuples:
                (stif_profile, num_hinges, kernel, spatial_kernel)
        seeds:
            Random seeds for each model configuration.
        sampling:
            Sampling method passed to run_pipeline_multiple_seeds().
        train_sample_fraction:
            Training fraction passed to run_pipeline_multiple_seeds().
        time_slices:
            Time slices for the overlaid rollout plot.
        q_growth_failure_threshold:
            Failure threshold used in run_pipeline_multiple_seeds().
        edge:
            Interior exclusion region for diagnostics.
        save_overlay_path:
            Path for the overlaid rollout plot.
        plot_overlay:
            If True, creates the overlaid rollout plot.
        print_table_fn:
            Optional function, e.g. print_metrics_table.

    Returns:
        results_by_label:
            Dictionary mapping short labels to multi-seed summaries.

        metrics_table:
            List of metric rows.

        model_summaries:
            List used for overlaid rollout plotting.
    """

    results_by_label = {}
    metrics_table = []
    model_summaries = []

    metric_keys = [
        "q_rel_l2",
        "u_rel_l2",
        "energy_rel_l2",
        "curl_rms",
        "dHdq_rmse",
        "dHdq_force_rmse",
    ]

    ref_test_data = None
    ref_simulator = None
    stiffness_profile_for_title = None

    for config in test_config:
        stif_profile, num_hinges, kernel, spatial_kernel = config

        # Avoid mutating the original param_dict across configurations.
        config_param_dict = copy.deepcopy(param_dict)
        config_param_dict["stiffness_profile"] = stif_profile
        config_param_dict["num_hinges"] = num_hinges
        config_param_dict["spatial_kernel"] = spatial_kernel

        label = get_caption_tag(kernel, config_param_dict)

        print("\n" + "#" * 90)
        print(
            f"CONFIG: {stif_profile} stiffness, "
            f"num_hinges={num_hinges}, "
            f"kernel={kernel}, "
            f"spatial_kernel={spatial_kernel}, "
            f"label={label}"
        )
        print("#" * 90)

        results = run_pipeline_multiple_seeds(
            param_dict=config_param_dict,
            seeds=seeds,
            kernel_type=kernel,
            train_sample_fraction=train_sample_fraction,
            sampling=sampling,
            make_individual_plots=False,

            # Disable per-config plots here.
            # The wrapper will make one clean overlaid rollout plot.
            plot_average_rollout=False,
            plot_average_error_curves=True,
            plot_average_heatmap=True,

            time_slices=time_slices,
            save_average_plot=True,
            q_growth_failure_threshold=q_growth_failure_threshold,
            edge=edge,
        )

        results_by_label[label] = results

        # --------------------------------------------------------
        # Store metric row
        # --------------------------------------------------------
        metric_row = [results["avg_metrics"][key] for key in metric_keys]
        metric_row.append(results["failure_stats"]["failure_rate"])
        metric_row.append(results["failure_stats"]["q_growth_ratio_mean"])

        metrics_table.append(metric_row)

        # --------------------------------------------------------
        # Store model summary for overlay plot
        # --------------------------------------------------------
        model_summaries.append(
            {
                "label": label,
                "q_gp_mean": results["q_gp_mean"],
                "avg_metrics": results["avg_metrics"],
                "failure_stats": results["failure_stats"],
            }
        )

        # Reference simulator/test data should be identical across configs
        # for a fixed stiffness profile. Use first config as reference.
        if ref_test_data is None:
            ref_test_data = results["all_outputs"][0]["test_data"]
            ref_simulator = results["all_outputs"][0]["simulator"]
            stiffness_profile_for_title = stif_profile

    print_metrics_table(metrics_table)

    # ------------------------------------------------------------
    # Overlaid rollout plot
    # ------------------------------------------------------------
    if plot_overlay:
        if save_overlay_path is None:
            save_overlay_path = (
                f"./plots/overlay_rollout_{stiffness_profile_for_title}_"
                f"nmodels{len(model_summaries)}_nseeds{len(seeds)}.png"
            )

        plot_overlaid_average_rollout_time_slices(
            data=ref_test_data,
            model_summaries=model_summaries,
            simulator=ref_simulator,
            time_slices=time_slices,
            save_path=save_overlay_path,
            title=f"Average Rollout Comparison ({stiffness_profile_for_title} stiffness)",
            legend_ncol=min(4, len(model_summaries) + 1),
        )

    return results_by_label, metrics_table, model_summaries


def run_diagnostic(param_dict, kernel_type="baseline"):
    sim = SoftRodSimulator(L=10.0, N=60, T=5.5, dt=0.05, num_hinges=param_dict["num_hinges"],
                           stiffness_profile=param_dict["stiffness_profile"])
    data = sim.generate_dataset(get_train_q_config)

    # p_flat = data['p'].flatten()
    # q_flat = data['q'].flatten()
    # dH_dq_target = data['dH_dq'].flatten()
    # dH_dp_target = data['dH_dp'].flatten()
    #
    # # Tile the spatial array to match the time dimension, then flatten
    # N_pts, T_steps = data['q'].shape
    # z_tiled = np.tile(data['z'].reshape(N_pts, 1), (1, T_steps))
    # z_flat = z_tiled.flatten()
    #
    # # X matrix is now shape [N_total, 3] -> (p, q, z)
    # X = np.stack([p_flat, q_flat, z_flat], axis=1)
    #
    # downscale_factor = 0.1
    # num_train = int(X.shape[0] * downscale_factor)
    # rng = np.random.default_rng(seed=42)
    # train_idx = rng.choice(X.shape[0], size=num_train, replace=False)
    #
    # X_train = torch.tensor(X[train_idx], dtype=torch.float64)
    # y_dEdp_train = torch.tensor(dH_dp_target[train_idx], dtype=torch.float64)
    # y_dEdq_train = torch.tensor(dH_dq_target[train_idx], dtype=torch.float64)
    #
    # # Z-Score Normalization
    # X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    # X_train_norm = (X_train - X_mean) / (X_std + 1e-6)
    #
    # y_p_mean, y_p_std = y_dEdp_train.mean(), y_dEdp_train.std()
    # y_p_train_norm = (y_dEdp_train - y_p_mean) / (y_p_std + 1e-6)
    #
    # y_q_mean, y_q_std = y_dEdq_train.mean(), y_dEdq_train.std()
    # y_q_train_norm = (y_dEdq_train - y_q_mean) / (y_q_std + 1e-6)
    #
    # dEdp_model, dEdp_lh = train_gp_model(X_train_norm, y_p_train_norm, param_dict, kernel_type="baseline",
    #                                      training_iter=1000, patience=15)
    # dEdq_model, dEdq_lh = train_gp_model(X_train_norm, y_q_train_norm, param_dict, kernel_type=kernel_type,
    #                                      training_iter=1000, patience=15)
    #
    # # Store normalization stats directly on the models for easy access in utils.py
    # dEdp_model.X_mean, dEdp_model.X_std = X_mean, X_std
    # dEdp_model.y_mean, dEdp_model.y_std = y_p_mean, y_p_std
    # dEdq_model.X_mean, dEdq_model.X_std = X_mean, X_std
    # dEdq_model.y_mean, dEdq_model.y_std = y_q_mean, y_q_std
    #
    # p_gp, q_gp = simulate_gp_phs(sim, dEdp_model, dEdq_model, get_test_q_config)
    # test_data = sim.generate_dataset(get_test_q_config)

    stiffness_activity_diagnostic(data, sim)
    # ph_force_diagnostic(data, sim)
    # gradient_error_by_z(data, sim, dEdq_model)
    # plot_q_dHdq_force_diagnostic(test_data, p_gp, q_gp, sim, dEdq_model, time_steps=(31, 32, 33))
    # plot_boundary_vs_interior_q(q_gp, test_data, sim)
    # scan_rollout_rhs_error(test_data, p_gp, q_gp, sim, dEdq_model, dpdt_bad_threshold=0.6)
    # for step in [30, 31, 32]:
    #     inspect_rhs_at_step(test_data, p_gp, q_gp, sim, dEdq_model, step)


if __name__ == '__main__':
    # ("periodic", 4, "SD", "rbf"), ("periodic", 4, "SD", "periodic"),
    # ("periodic", 4, "SD", "warped"), ("periodic", 4, "warped", "rbf")
    # ("C2", 8, "baseline", "rbf"), ("C2", 8, "SD", "rbf"), ("C2", 8, "SD", "matern")
    # ("piecewise", 2, "baseline", "rbf"), ("piecewise", 2, "SD", "rbf"),
    # ("piecewise", 2, "SD", "matern"), ("piecewise", 2, "SD", "changepoint")
    test_config = [("piecewise", 2, "baseline", "rbf"), ("piecewise", 2, "SD", "rbf"),
                   ("piecewise", 2, "SD", "matern"), ("piecewise", 2, "SD", "changepoint")]
    param_dict = {
        "stiffness_profile": "piecewise",
        "num_hinges": 2,
        "strain_kernel": "rbf",
        "spatial_kernel": "changepoint",
        "matern_nu": 2.5,
        "changepoint_location": 5.0,
        "changepoint_width": 0.15,
        "n_freqs": 10
    }
    run_model_configurations_with_overlay(param_dict, test_config, seeds=(6, 42, 79, 123, 456),
                                          sampling="stratified", train_sample_fraction=0.04)
    # metrics_table = []
    # for stif_profile, num_hinges, kernel, spatial_kernel in test_config:
    #     print(f"CONFIG: {stif_profile} stiffness profile, {num_hinges} bumps/hinges, {kernel} kernel, {spatial_kernel} spatial kernel")
    #     param_dict["stiffness_profile"] = stif_profile
    #     param_dict["num_hinges"] = num_hinges
    #     param_dict["spatial_kernel"] = spatial_kernel
    #     results = run_pipeline_multiple_seeds(param_dict, seeds=(6, 42, 79, 123, 456), kernel_type=kernel,
    #                             sampling="stratified", train_sample_fraction=0.06)
    #     metric_summary = [results["avg_metrics"][key] for key in results["avg_metrics"]]
    #     metric_summary.append(results["failure_stats"]["failure_rate"])
    #     metric_summary.append(results["failure_stats"]["q_growth_ratio_mean"])
    #     metrics_table.append(metric_summary)
    #
    # print_metrics_table(metrics_table)

    # run_pipeline_multiple_seeds(param_dict, seeds=(6, 42, 79, 123, 456), kernel_type="baseline",
    #                             sampling="stratified", train_sample_fraction=0.06)
    # run_pipeline(param_dict, kernel_type="SD", sampling="stratified", seed=79,
    #              train_sample_fraction=0.04, verbose=True, make_plots=True)
    # run_diagnostic(param_dict)