from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import gpytorch
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid, solve_ivp

from config import InitialCondition, NormalizationStats, TrainedModels


class PhysicalMeanPredictor(torch.nn.Module):
    """
    Evaluate one ExactGP posterior mean and transform it back to physical units.

    The normalization tensors are prepared once rather than transferred or
    retrieved repeatedly during every ODE right-hand-side evaluation.
    """

    def __init__(
        self,
        model: Any,
        reference_input: torch.Tensor,
    ) -> None:
        super().__init__()

        self.model = model

        device = reference_input.device

        X_mean = cast(torch.Tensor, model.X_mean)
        X_std = cast(torch.Tensor, model.X_std)

        y_mean = cast(torch.Tensor, model.y_mean)
        y_std = cast(torch.Tensor, model.y_std)

        # Match the existing code: move to the input device but do not
        # explicitly change dtype.
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

    def forward(
        self,
        X_raw: torch.Tensor,
    ) -> torch.Tensor:
        X_normalized = (X_raw - self.X_mean) / (self.X_std + 1e-6)
        posterior = self.model(X_normalized)
        posterior_mean = cast(torch.Tensor, posterior.mean)

        return posterior_mean * self.y_std + self.y_mean


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
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.cpu()
    return detached.numpy()


MeanPredictor = Callable[
    [torch.Tensor],
    torch.Tensor,
]


def prepare_mean_predictor(
    model: Any,
    example_input: torch.Tensor,
    use_torch_compile: bool,
) -> MeanPredictor:
    """
    Construct and optionally compile a fixed-shape physical-mean predictor.

    The eager call initializes the ExactGP prediction cache before compilation.
    The first compiled call triggers compilation and verifies that its output
    agrees with the eager implementation.
    """
    model.eval()

    predictor = PhysicalMeanPredictor(
        model=model,
        reference_input=example_input,
    ).eval()

    # This first call also initializes GPyTorch's ExactGP prediction strategy.
    eager_output = predictor(example_input).detach().clone()

    if not use_torch_compile:
        return predictor

    if not hasattr(torch, "compile"):
        warnings.warn(
            "torch.compile is unavailable in this PyTorch version. "
            "Using the eager GP predictor.",
            RuntimeWarning,
            stacklevel=2,
        )

        return predictor

    try:
        compiled_predictor = torch.compile(
            predictor,
            dynamic=False,
            fullgraph=False,
        )

        # Compilation normally occurs on the first invocation.
        compiled_output = compiled_predictor(example_input)

        torch.testing.assert_close(
            compiled_output,
            eager_output,
            rtol=1e-6,
            atol=1e-8,
        )

        return compiled_predictor

    except Exception as error:
        warnings.warn(
            "torch.compile could not compile the GP predictor; "
            "falling back to eager inference. "
            f"Reason: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

        return predictor


def simulate_gp_phs(
    simulator: Any,
    models: TrainedModels,
    initial_condition: InitialCondition,
    compile_gp_predictors: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the learned GP-PHS dynamics.

    Optimizations:
        1. Preallocate the NumPy input matrix.
        2. Share it with Torch through torch.from_numpy().
        3. Prepare normalization tensors once.
        4. Optionally compile the two physical-mean predictors.
        5. Disable gradient tracking around the complete IVP solve.
        6. Skip posterior covariance construction because only the mean is used.
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

    # The zero-copy NumPy/Torch buffer path is designed for CPU inference,
    # which matches the current SciPy-based rollout.
    if dEdp_device.type != "cpu":
        raise ValueError(
            "The optimized SciPy rollout currently requires CPU GP models. "
            "A GPU model would require copying every RK45 input and output "
            "between CPU and GPU."
        )

    # Preallocate one input matrix for the complete rollout.
    #
    # NumPy owns the memory. torch.from_numpy() creates a tensor that shares
    # that memory without copying it.
    X_raw_numpy = np.empty((simulator.N, 3), dtype=np.float64)

    X_raw_numpy[:, 0] = p0
    X_raw_numpy[:, 1] = q0
    X_raw_numpy[:, 2] = np.asarray(simulator.z, dtype=np.float64)

    X_raw_tensor = torch.from_numpy(X_raw_numpy)

    t_eval = np.arange(0.0, simulator.T, simulator.dt, dtype=np.float64)

    solver_config = simulator.config.ode_solver

    # Put the context around predictor preparation and the entire synchronous
    # solve rather than entering torch.no_grad() inside every RHS evaluation.
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        predict_dHdp = prepare_mean_predictor(
            model=models.dEdp_model,
            example_input=X_raw_tensor,
            use_torch_compile=compile_gp_predictors,
        )

        predict_dHdq = prepare_mean_predictor(
            model=models.dEdq_model,
            example_input=X_raw_tensor,
            use_torch_compile=compile_gp_predictors,
        )

        def gp_pde_rhs(
            t: float,
            state: np.ndarray,
        ) -> np.ndarray:
            p = state[: simulator.N]
            q = state[simulator.N :]

            # Update only the two state-dependent columns.
            # The z column was filled once before solve_ivp().
            X_raw_numpy[:, 0] = p
            X_raw_numpy[:, 1] = q

            grad_H_p = tensor_to_numpy(predict_dHdp(X_raw_tensor))
            grad_H_q = tensor_to_numpy(predict_dHdq(X_raw_tensor))

            return simulator.evaluate_ph_dynamics(
                p=p,
                q=q,
                grad_H_p=grad_H_p,
                grad_H_q=grad_H_q,
                t=t,
            )

        solution = solve_ivp(
            fun=gp_pde_rhs,
            t_span=(
                0.0,
                float(simulator.T),
            ),
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
    """Reconstruct u with u(0,t)=0 using the project's existing scheme."""

    q = np.asarray(q, dtype=float)

    if q.ndim != 2:
        raise ValueError("q must have shape (N_z, N_t).")

    return cumulative_trapezoid(
        q,
        dx=dz,
        axis=0,
        initial=0,
    )
