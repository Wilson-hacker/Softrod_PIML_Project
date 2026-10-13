from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

import gpytorch
import numpy as np
import torch

from config import (
    AggregateResult,
    ChangepointEstimate,
    DatasetBundle,
    EvaluationConfig,
    RunResult,
    TrainedModels,
)
from rollout import (
    PhysicalMeanPredictor,
    reconstruct_displacement_from_strain,
    spatial_derivative_centered,
    tensor_to_numpy,
)

SELECTED_METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "q_rel_l2": ("rollout", "q_rel_l2"),
    "u_rel_l2": ("rollout", "u_rel_l2"),
    "energy_rel_l2": ("energy", "energy_rel_l2"),
    "curl_rms": ("integrability", "curl_rms"),
    "dHdq_rmse": ("calibration", "dHdq", "rmse"),
    "dHdq_force_rmse": (
        "calibration",
        "dHdq_force",
        "dHdq_force_rmse",
    ),
}


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return the L2 error relative to the reference norm."""
    denominator = np.linalg.norm(reference.reshape(-1)) + 1e-12
    return float(np.linalg.norm((prediction - reference).reshape(-1)) / denominator)


def rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def mae(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(np.mean(np.abs(prediction - reference)))


def aligned_trajectories(
    data: Mapping[str, np.ndarray],
    p_gp: np.ndarray,
    q_gp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trim reference and predicted trajectories to a common time range."""
    p_reference = np.asarray(data["p"])
    q_reference = np.asarray(data["q"])
    n_t = min(
        p_reference.shape[1],
        q_reference.shape[1],
        p_gp.shape[1],
        q_gp.shape[1],
    )
    return (
        p_reference[:, :n_t],
        q_reference[:, :n_t],
        p_gp[:, :n_t],
        q_gp[:, :n_t],
    )


def compute_rollout_error_metrics(
    data: Mapping[str, np.ndarray],
    p_gp: np.ndarray,
    q_gp: np.ndarray,
    simulator: Any,
) -> dict[str, float]:
    """Measure errors in momentum, strain, and reconstructed displacement."""
    p_ref, q_ref, p_gp, q_gp = aligned_trajectories(data, p_gp, q_gp)
    state_ref = np.concatenate((p_ref, q_ref), axis=0)
    state_gp = np.concatenate((p_gp, q_gp), axis=0)
    u_ref = reconstruct_displacement_from_strain(q_ref, simulator.dz)
    u_gp = reconstruct_displacement_from_strain(q_gp, simulator.dz)

    return {
        "p_rel_l2": relative_l2(p_gp, p_ref),
        "q_rel_l2": relative_l2(q_gp, q_ref),
        "state_rel_l2": relative_l2(state_gp, state_ref),
        "u_rel_l2": relative_l2(u_gp, u_ref),
        "p_rmse": rmse(p_gp, p_ref),
        "q_rmse": rmse(q_gp, q_ref),
        "state_rmse": rmse(state_gp, state_ref),
        "u_rmse": rmse(u_gp, u_ref),
        "p_final_rel_l2": relative_l2(p_gp[:, -1], p_ref[:, -1]),
        "q_final_rel_l2": relative_l2(q_gp[:, -1], q_ref[:, -1]),
        "u_final_rel_l2": relative_l2(u_gp[:, -1], u_ref[:, -1]),
    }


def compute_energy_error_metrics(
    data: Mapping[str, np.ndarray],
    p_gp: np.ndarray,
    q_gp: np.ndarray,
    simulator: Any,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Compare Hamiltonian values along reference and GP trajectories."""
    p_ref, q_ref, p_gp, q_gp = aligned_trajectories(data, p_gp, q_gp)
    H_reference = simulator.total_energy(p_ref, q_ref)
    H_gp = simulator.total_energy(p_gp, q_gp)

    metrics = {
        "energy_rel_l2": relative_l2(H_gp, H_reference),
        "energy_rmse": rmse(H_gp, H_reference),
        "energy_mae": mae(H_gp, H_reference),
        "energy_final_abs_error": float(abs(H_gp[-1] - H_reference[-1])),
        "energy_final_rel_error": float(
            abs(H_gp[-1] - H_reference[-1]) / (abs(H_reference[-1]) + 1e-12)
        ),
    }
    return metrics, H_reference, H_gp


def compute_curl_integrability_error(
    dEdp_model: Any,
    dEdq_model: Any,
    X_raw: np.ndarray,
    max_points: int = 2048,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, float]:
    """Measure curl = d/dq(dH/dp) - d/dp(dH/dq) in physical units."""
    dEdp_model.eval()
    dEdq_model.eval()
    rng = np.random.default_rng(seed)

    model_device = dEdp_model.X_mean.device
    model_dtype = dEdp_model.X_mean.dtype

    if X_raw.shape[0] > max_points:
        indices = rng.choice(X_raw.shape[0], size=max_points, replace=False)
        X_eval = X_raw[indices]
    else:
        X_eval = X_raw

    reference_input = torch.as_tensor(
        X_eval[: min(batch_size, X_eval.shape[0])],
        device=model_device,
        dtype=model_dtype,
    )
    dEdp_predictor = PhysicalMeanPredictor(
        model=dEdp_model, reference_input=reference_input
    ).eval()
    dEdq_predictor = PhysicalMeanPredictor(
        model=dEdq_model, reference_input=reference_input
    ).eval()

    curl_batches: list[np.ndarray] = []

    with torch.enable_grad():
        for start in range(0, X_eval.shape[0], batch_size):
            X_batch = torch.as_tensor(
                X_eval[start : start + batch_size],
                device=model_device,
                dtype=model_dtype,
            )

            X_p = X_batch.clone().detach().requires_grad_(True)
            mean_p = dEdp_predictor(X_p)
            gradient_p = torch.autograd.grad(mean_p.sum(), X_p)[0]

            X_q = X_batch.clone().detach().requires_grad_(True)
            mean_q = dEdq_predictor(X_q)
            gradient_q = torch.autograd.grad(mean_q.sum(), X_q)[0]

            curl_batches.append(tensor_to_numpy(gradient_p[:, 1] - gradient_q[:, 0]))

    curl = np.concatenate(curl_batches)
    return {
        "curl_mean_abs": float(np.mean(np.abs(curl))),
        "curl_rms": float(np.sqrt(np.mean(curl**2))),
        "curl_max_abs": float(np.max(np.abs(curl))),
    }


def gp_predict_with_uncertainty(
    model: Any,
    likelihood: Any,
    X_raw: np.ndarray,
    batch_size: int = 2048,
    include_likelihood_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict means and standard deviations in physical units."""
    model.eval()
    likelihood.eval()
    model_device = model.X_mean.device
    model_dtype = model.X_mean.dtype

    reference_input = torch.as_tensor(X_raw[:1], device=model_device, dtype=model_dtype)
    predictor = PhysicalMeanPredictor(
        model=model, reference_input=reference_input
    ).eval()

    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, X_raw.shape[0], batch_size):
            X_batch = torch.as_tensor(
                X_raw[start : start + batch_size],
                device=model_device,
                dtype=model_dtype,
            )
            latent_distribution = predictor.latent_distribution(X_batch)
            distribution = (
                likelihood(latent_distribution)
                if include_likelihood_noise
                else latent_distribution
            )

            normalized_mean = distribution.mean
            normalized_variance = distribution.variance.clamp_min(1.0e-12)

            physical_mean = predictor.denormalize_mean(normalized_mean)
            physical_standard_deviation = predictor.denormalize_std(
                torch.sqrt(normalized_variance)
            )

            means.append(tensor_to_numpy(physical_mean))
            standard_deviations.append(tensor_to_numpy(physical_standard_deviation))

    return np.concatenate(means), np.concatenate(standard_deviations)


def compute_single_calibration_metrics(
    y_reference: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> dict[str, float]:
    """Compute error and uncertainty calibration metrics for one GP output."""
    standard_deviation = np.maximum(standard_deviation, 1e-12)
    residual = y_reference - mean
    z_score = residual / standard_deviation
    nlpd = 0.5 * np.mean(
        z_score**2 + 2.0 * np.log(standard_deviation) + np.log(2.0 * np.pi)
    )

    return {
        "rmse": rmse(mean, y_reference),
        "mae": mae(mean, y_reference),
        "avg_std": float(np.mean(standard_deviation)),
        "coverage_68": float(np.mean(np.abs(z_score) <= 1.0)),
        "coverage_95": float(np.mean(np.abs(z_score) <= 1.96)),
        "mean_abs_z": float(np.mean(np.abs(z_score))),
        "nlpd": float(nlpd),
    }


def compute_dHdq_force_metrics_from_predictions(
    dHdq_prediction_flat: np.ndarray,
    dHdq_reference_flat: np.ndarray,
    simulator: Any,
    grid_shape: tuple[int, int],
    edge: int = 5,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return force-error metrics and fields from flattened effort predictions."""
    n_z, n_t = grid_shape
    dHdq_prediction = np.asarray(dHdq_prediction_flat).reshape(n_z, n_t)
    dHdq_reference = np.asarray(dHdq_reference_flat).reshape(n_z, n_t)

    force_prediction = np.full_like(dHdq_prediction, np.nan, dtype=float)
    force_reference = np.full_like(dHdq_reference, np.nan, dtype=float)

    for time_index in range(n_t):
        force_prediction[:, time_index] = spatial_derivative_centered(
            dHdq_prediction[:, time_index], simulator.dz
        )
        force_reference[:, time_index] = spatial_derivative_centered(
            dHdq_reference[:, time_index], simulator.dz
        )

    interior = slice(edge, n_z - edge)
    force_prediction_int = force_prediction[interior]
    force_reference_int = force_reference[interior]
    finite_mask = np.isfinite(force_prediction_int - force_reference_int)
    force_rmse = rmse(
        force_prediction_int[finite_mask],
        force_reference_int[finite_mask],
    )
    force_reference_rms = float(np.sqrt(np.nanmean(force_reference_int**2)) + 1e-12)

    return (
        {
            "dHdq_force_rmse": force_rmse,
            "dHdq_force_relative_rmse": force_rmse / force_reference_rms,
            "dHdq_force_true_rms_scale": force_reference_rms,
            "dHdq_force_max_abs_error": float(
                np.nanmax(np.abs(force_prediction - force_reference))
            ),
        },
        {
            "dHdq_force_pred": force_prediction,
            "dHdq_force_true": force_reference,
            "dHdq_force_error": force_prediction - force_reference,
        },
    )


def compute_uncertainty_calibration_metrics(
    models: TrainedModels,
    dataset: DatasetBundle,
    simulator: Any,
    evaluation_config: EvaluationConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate held-out effort calibration and full-grid force error."""
    dHdp_mean, dHdp_std = gp_predict_with_uncertainty(
        models.dEdp_model,
        models.dEdp_likelihood,
        dataset.X_all,
        include_likelihood_noise=evaluation_config.include_likelihood_noise,
    )
    dHdq_mean, dHdq_std = gp_predict_with_uncertainty(
        models.dEdq_model,
        models.dEdq_likelihood,
        dataset.X_all,
        include_likelihood_noise=evaluation_config.include_likelihood_noise,
    )

    heldout = dataset.heldout_indices
    metrics: dict[str, Any] = {
        "dHdp": compute_single_calibration_metrics(
            dataset.y_p_all[heldout],
            dHdp_mean[heldout],
            dHdp_std[heldout],
        ),
        "dHdq": compute_single_calibration_metrics(
            dataset.y_q_all[heldout],
            dHdq_mean[heldout],
            dHdq_std[heldout],
        ),
    }

    force_metrics, dHdq_summary = compute_dHdq_force_metrics_from_predictions(
        dHdq_prediction_flat=dHdq_mean,
        dHdq_reference_flat=dataset.y_q_all,
        simulator=simulator,
        grid_shape=dataset.grid_shape,
        edge=evaluation_config.force_edge,
    )
    metrics["dHdq_force"] = force_metrics
    dHdq_summary.update(
        {
            "dHdq_mean": dHdq_mean,
            "dHdq_true": dataset.y_q_all,
            "dHdq_std": dHdq_std,
        }
    )
    return metrics, dHdq_summary


def evaluate_gp_phs_model(
    evaluation_data: Mapping[str, np.ndarray],
    simulator: Any,
    p_gp: np.ndarray,
    q_gp: np.ndarray,
    models: TrainedModels,
    dataset: DatasetBundle,
    evaluation_config: EvaluationConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Evaluate rollout, energy, integrability, and calibration errors."""
    rollout_metrics = compute_rollout_error_metrics(
        evaluation_data, p_gp, q_gp, simulator
    )
    energy_metrics, H_reference, H_gp = compute_energy_error_metrics(
        evaluation_data, p_gp, q_gp, simulator
    )
    curl_metrics = compute_curl_integrability_error(
        models.dEdp_model,
        models.dEdq_model,
        dataset.X_all,
        max_points=evaluation_config.curl_max_points,
        batch_size=evaluation_config.curl_batch_size,
        seed=seed,
    )
    calibration_metrics, dHdq_summary = compute_uncertainty_calibration_metrics(
        models,
        dataset,
        simulator,
        evaluation_config,
    )

    metrics = {
        "rollout": rollout_metrics,
        "energy": energy_metrics,
        "integrability": curl_metrics,
        "calibration": calibration_metrics,
    }
    return metrics, dHdq_summary, H_reference, H_gp


def get_nested_metric(metrics: Mapping[str, Any], path: Sequence[str]) -> float:
    """Read a scalar metric from a nested dictionary path."""
    current: Any = metrics
    for key in path:
        current = current[key]
    return float(current)


def extract_selected_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Collect the scalar metrics used in the aggregate report."""
    return {
        name: get_nested_metric(metrics, path)
        for name, path in SELECTED_METRIC_PATHS.items()
    }


def aggregate_scalar_values(
    values: Sequence[float],
) -> tuple[float, float]:
    """Return the mean and sample standard deviation, or zero for one sample."""
    values_array = np.asarray(values, dtype=float)
    mean = float(values_array.mean())
    standard_deviation = (
        float(values_array.std(ddof=1)) if values_array.size > 1 else 0.0
    )

    return mean, standard_deviation


def aggregate_run_results(runs: Sequence[RunResult]) -> AggregateResult:
    """Aggregate metrics, trajectories, and changepoint estimates across seeds."""
    if not runs:
        raise ValueError("At least one run is required for aggregation.")

    first = runs[0]
    for run in runs[1:]:
        if run.experiment_name != first.experiment_name:
            raise ValueError("Cannot aggregate different experiments.")
        if run.model_label != first.model_label:
            raise ValueError("Cannot aggregate different model labels.")

    selected = [extract_selected_metrics(run.metrics) for run in runs]
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}

    for key in SELECTED_METRIC_PATHS:
        values = np.asarray([entry[key] for entry in selected], dtype=float)
        mean_metrics[key], std_metrics[key] = aggregate_scalar_values(values)

    p_stack = np.stack([run.p_gp for run in runs], axis=0)
    q_stack = np.stack([run.q_gp for run in runs], axis=0)

    changepoints = [run.changepoint for run in runs]

    if all(changepoint is None for changepoint in changepoints):
        changepoint_mean = None
        changepoint_std = None

    elif any(changepoint is None for changepoint in changepoints):
        raise ValueError(
            "Some seeds contain learned changepoint parameters while others do not."
        )

    else:
        valid_changepoints = [
            cast(ChangepointEstimate, changepoint) for changepoint in changepoints
        ]
        center_mean, center_std = aggregate_scalar_values(
            [changepoint.center for changepoint in valid_changepoints]
        )
        width_mean, width_std = aggregate_scalar_values(
            [changepoint.width for changepoint in valid_changepoints]
        )

        changepoint_mean = ChangepointEstimate(center=center_mean, width=width_mean)
        changepoint_std = ChangepointEstimate(center=center_std, width=width_std)

    return AggregateResult(
        experiment_name=first.experiment_name,
        model_label=first.model_label,
        runs=tuple(runs),
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        p_gp_mean=p_stack.mean(axis=0),
        p_gp_std=p_stack.std(axis=0, ddof=1)
        if len(runs) > 1
        else np.zeros_like(p_stack[0]),
        q_gp_mean=q_stack.mean(axis=0),
        q_gp_std=q_stack.std(axis=0, ddof=1)
        if len(runs) > 1
        else np.zeros_like(q_stack[0]),
        changepoint_mean=changepoint_mean,
        changepoint_std=changepoint_std,
    )


def print_metrics_report(metrics: Mapping[str, Any]) -> None:
    """Print the evaluation metrics."""
    print("\n================ GP-PHS Evaluation Metrics ================\n")
    for section_key, title in (
        ("rollout", "Rollout Error"),
        ("energy", "Energy Error"),
        ("integrability", "Curl / Integrability Error"),
    ):
        print(f"[{title}]")
        for metric_name, value in metrics[section_key].items():
            print(f"  {metric_name:32s}: {value:.6e}")
        print()

    print("[Effort Calibration: dH/dq]")
    for metric_name, value in metrics["calibration"]["dHdq"].items():
        print(f"  {metric_name:32s}: {value:.6e}")

    print("\n[Force-Field Error: d/dz(dH/dq)]")
    for metric_name, value in metrics["calibration"]["dHdq_force"].items():
        print(f"  {metric_name:32s}: {value:.6e}")

    print("\n============================================================\n")
