from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp, cumulative_trapezoid

from config import InitialCondition, NormalizationStats, TrainedModels


def attach_normalization_stats(
    models: TrainedModels,
    normalization: NormalizationStats,
) -> None:
    """Attach normalization metadata expected by the current GP models."""
    models.dEdp_model.X_mean = normalization.X_mean
    models.dEdp_model.X_std = normalization.X_std
    models.dEdp_model.y_mean = normalization.y_p_mean
    models.dEdp_model.y_std = normalization.y_p_std

    models.dEdq_model.X_mean = normalization.X_mean
    models.dEdq_model.X_std = normalization.X_std
    models.dEdq_model.y_mean = normalization.y_q_mean
    models.dEdq_model.y_std = normalization.y_q_std


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def model_mean_physical(model: Any, X_raw_tensor: torch.Tensor) -> torch.Tensor:
    """Evaluate a GP posterior mean and transform it back to physical units."""
    X_mean = model.X_mean.to(X_raw_tensor.device)
    X_std = model.X_std.to(X_raw_tensor.device)
    y_mean = model.y_mean.to(X_raw_tensor.device)
    y_std = model.y_std.to(X_raw_tensor.device)

    X_norm = (X_raw_tensor - X_mean) / (X_std + 1e-6)
    return model(X_norm).mean * y_std + y_mean


def simulate_gp_phs(
    simulator: Any,
    models: TrainedModels,
    initial_condition: InitialCondition,
    method: str = "RK45",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the learned GP-PHS dynamics from the supplied initial state."""
    q0 = np.asarray(initial_condition(simulator.z, simulator.L), dtype=float)
    p0 = np.zeros(simulator.N, dtype=float)
    state0 = np.concatenate((p0, q0))

    models.dEdp_model.eval()
    models.dEdq_model.eval()

    def gp_pde_rhs(t: float, state: np.ndarray) -> np.ndarray:
        p = state[: simulator.N]
        q = state[simulator.N :]
        X_raw = torch.as_tensor(
            np.column_stack((p, q, simulator.z)),
            dtype=torch.float64,
        )

        with torch.no_grad():
            grad_H_p = tensor_to_numpy(model_mean_physical(models.dEdp_model, X_raw))
            grad_H_q = tensor_to_numpy(model_mean_physical(models.dEdq_model, X_raw))

        return simulator.evaluate_ph_dynamics(p, q, grad_H_p, grad_H_q, t)

    t_eval = np.arange(0.0, simulator.T, simulator.dt)
    solution = solve_ivp(
        gp_pde_rhs,
        (0.0, simulator.T),
        state0,
        t_eval=t_eval,
        method=method,
    )

    if not solution.success:
        raise RuntimeError(f"GP-PHS rollout failed: {solution.message}")
    if solution.y.shape[1] != t_eval.size:
        raise RuntimeError("GP-PHS rollout returned fewer output times than requested.")

    return (
        solution.y[: simulator.N, :],
        solution.y[simulator.N :, :],
    )


def spatial_derivative_centered(
    field: np.ndarray,
    dz: float,
    boundary_value: float = np.nan,
) -> np.ndarray:
    """Apply the simulator's centered spatial-difference stencil."""
    derivative = np.full_like(field, boundary_value, dtype=float)
    derivative[1:-1] = (field[2:] - field[:-2]) / (2.0 * dz)
    return derivative


def reconstruct_displacement_from_strain(
    q: np.ndarray,
    dz_or_simulator: float | Any,
) -> np.ndarray:
    """Reconstruct u with u(0,t)=0 using the project's existing scheme."""
    dz = (
        float(dz_or_simulator)
        if np.isscalar(dz_or_simulator)
        else float(dz_or_simulator.dz)
    )

    q = np.asarray(q, dtype=float)

    if q.ndim != 2:
        raise ValueError(
            "q must have shape (N_z, N_t)."
        )

    return cumulative_trapezoid(
        q,
        dx=dz,
        axis=0,
        initial=0.0,
    )
    # return np.cumsum(q, axis=0) * dz
