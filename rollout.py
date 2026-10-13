from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import gpytorch
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid, solve_ivp

from config import (
    InitialCondition,
    NormalizationStats,
    TrainedModels,
)
from data_processing import (
    denormalize_standard_deviation,
    denormalize_tensor,
    normalize_tensor,
)


class PhysicalMeanPredictor(torch.nn.Module):
    """Evaluate GP posterior means in physical units using cached normalization."""

    X_mean: torch.Tensor
    X_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def __init__(
        self,
        model: Any,
        reference_input: torch.Tensor,
    ) -> None:
        """Cache the model normalization statistics on the input device."""
        super().__init__()

        self.model = model

        device = reference_input.device

        X_mean = cast(torch.Tensor, model.X_mean)
        X_std = cast(torch.Tensor, model.X_std)

        y_mean = cast(torch.Tensor, model.y_mean)
        y_std = cast(torch.Tensor, model.y_std)

        # Move statistics to the input device while preserving their dtype.
        self.register_buffer(
            "X_mean",
            X_mean.detach().to(device=device).clone(),
        )
        self.register_buffer(
            "X_std",
            X_std.detach().to(device=device).clone(),
        )

        self.register_buffer(
            "y_mean",
            y_mean.detach().to(device=device).clone(),
        )
        self.register_buffer(
            "y_std",
            y_std.detach().to(device=device).clone(),
        )

    def normalize_inputs(
        self,
        X_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Standardize physical inputs using the cached statistics."""
        return normalize_tensor(
            values=X_raw, mean=self.X_mean, standard_deviation=self.X_std
        )

    def latent_distribution(
        self,
        X_raw: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        """Return the latent GP posterior for physical inputs."""
        X_normalized = self.normalize_inputs(X_raw)

        distribution = self.model(X_normalized)

        return cast(gpytorch.distributions.MultivariateNormal, distribution)

    def denormalize_mean(
        self,
        normalized_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a normalized posterior mean to physical units."""
        return denormalize_tensor(
            normalized_values=normalized_mean,
            mean=self.y_mean,
            standard_deviation=self.y_std,
        )

    def denormalize_std(
        self,
        normalized_standard_deviation: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a normalized posterior standard deviation to physical units."""
        return denormalize_standard_deviation(
            normalized_standard_deviation=normalized_standard_deviation,
            standard_deviation=self.y_std,
        )

    def forward(
        self,
        X_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the posterior mean in physical units."""
        distribution = self.latent_distribution(X_raw)
        normalized_mean = distribution.mean
        return self.denormalize_mean(normalized_mean)


def attach_normalization_stats(
    models: TrainedModels,
    normalization: NormalizationStats,
) -> None:
    """Attach input and output normalization statistics to both GP models."""
    models.dEdp_model.X_mean = normalization.X_mean
    models.dEdp_model.X_std = normalization.X_std
    models.dEdp_model.y_mean = normalization.y_p_mean
    models.dEdp_model.y_std = normalization.y_p_std

    models.dEdq_model.X_mean = normalization.X_mean
    models.dEdq_model.X_std = normalization.X_std
    models.dEdq_model.y_mean = normalization.y_q_mean
    models.dEdq_model.y_std = normalization.y_q_std


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and convert it to a NumPy array on CPU."""
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.cpu()
    return detached.numpy()


MeanPredictor = Callable[
    [torch.Tensor],
    torch.Tensor,
]


def simulate_gp_phs(
    simulator: Any,
    models: TrainedModels,
    initial_condition: InitialCondition,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the learned GP-PHS dynamics on CPU.

    Reuse a shared NumPy/Torch input buffer and cached normalization tensors.
    Disable gradients and posterior variances during integration.
    """
    q0 = np.asarray(initial_condition(simulator.z, simulator.L), dtype=np.float64)

    if q0.shape != (simulator.N,):
        raise ValueError(
            "The initial strain must have shape "
            f"({simulator.N},), but received {q0.shape}."
        )

    p0 = np.zeros(simulator.N, dtype=np.float64)

    state0 = np.concatenate((p0, q0))

    models.dEdp_model.eval()
    models.dEdq_model.eval()

    dEdp_device = next(models.dEdp_model.parameters()).device

    dEdq_device = next(models.dEdq_model.parameters()).device

    if dEdp_device != dEdq_device:
        raise ValueError("The dH/dp and dH/dq models must be on the same device.")

    # Sharing the NumPy/Torch input buffer requires CPU models.
    if dEdp_device.type != "cpu":
        raise ValueError(
            "The optimized SciPy rollout currently requires CPU GP models. "
            "A GPU model would require copying every RK45 input and output "
            "between CPU and GPU."
        )

    # Reuse one input buffer shared by NumPy and Torch.
    X_raw_numpy = np.empty((simulator.N, 3), dtype=np.float64)

    X_raw_numpy[:, 0] = p0
    X_raw_numpy[:, 1] = q0
    X_raw_numpy[:, 2] = np.asarray(simulator.z, dtype=np.float64)

    X_raw_tensor = torch.from_numpy(X_raw_numpy)

    t_eval = np.arange(0.0, simulator.T, simulator.dt, dtype=np.float64)

    solver_config = simulator.config.ode_solver

    # Disable gradients and posterior variances for the full solve.
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        models.dEdp_model.eval()
        models.dEdq_model.eval()
        predict_dHdp = PhysicalMeanPredictor(
            model=models.dEdp_model, reference_input=X_raw_tensor
        ).eval()
        predict_dHdq = PhysicalMeanPredictor(
            model=models.dEdq_model, reference_input=X_raw_tensor
        ).eval()

        def gp_pde_rhs(
            t: float,
            state: np.ndarray,
        ) -> np.ndarray:
            """Evaluate the dynamics using GP-predicted effort fields."""
            p = state[: simulator.N]
            q = state[simulator.N :]

            # Update p and q; the spatial coordinates remain fixed.
            X_raw_numpy[:, 0] = p
            X_raw_numpy[:, 1] = q

            grad_H_p = tensor_to_numpy(predict_dHdp(X_raw_tensor))
            grad_H_q = tensor_to_numpy(predict_dHdq(X_raw_tensor))

            return simulator.evaluate_ph_dynamics(
                grad_H_p=grad_H_p, grad_H_q=grad_H_q, t=t
            )

        solution = solve_ivp(
            fun=gp_pde_rhs,
            t_span=(0.0, float(simulator.T)),
            y0=state0,
            t_eval=t_eval,
            method=solver_config.method,
            rtol=solver_config.rtol,
            atol=solver_config.atol,
            max_step=(
                np.inf if solver_config.max_step is None else solver_config.max_step
            ),
        )

    if not solution.success:
        raise RuntimeError(f"GP-PHS rollout failed: {solution.message}")

    if solution.y.shape[1] != t_eval.size:
        raise RuntimeError("GP-PHS rollout returned fewer output times than requested.")

    return solution.y[: simulator.N, :], solution.y[simulator.N :, :]


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
    dz: float,
) -> np.ndarray:
    """Integrate strain with the trapezoidal rule and u(0, t) = 0."""

    q = np.asarray(q, dtype=float)

    if q.ndim != 2:
        raise ValueError("q must have shape (N_z, N_t).")

    return cumulative_trapezoid(q, dx=dz, axis=0, initial=0)
