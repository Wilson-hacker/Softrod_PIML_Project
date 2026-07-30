from __future__ import annotations

import copy
import math
from typing import cast

import gpytorch
import torch

from config import ModelConfig, TrainingConfig
from kernels import build_kernel


class EnergyGradientGP(gpytorch.models.ExactGP):
    """Exact GP model for one Hamiltonian effort component."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        model_config: ModelConfig,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = build_kernel(model_config)

    def forward(self, x: torch.Tensor):
        mean = cast(torch.Tensor, self.mean_module(x))
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


def train_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> tuple[
    EnergyGradientGP,
    gpytorch.likelihoods.GaussianLikelihood,
]:
    """Train one exact GP with early stopping and best-state restoration."""
    if train_x.ndim != 2:
        raise ValueError("train_x must have shape (N, D).")
    if train_y.ndim != 1:
        raise ValueError("train_y must have shape (N,).")
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("train_x and train_y must contain the same samples.")
    if not torch.isfinite(train_x).all() or not torch.isfinite(train_y).all():
        raise ValueError("Training data contains nonfinite values.")

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=train_x.device)
    model = EnergyGradientGP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        model_config=model_config,
    ).to(device=train_x.device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
    )
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood,
        model,
    )

    best_loss = math.inf
    best_iteration = -1
    best_model_state = None
    best_likelihood_state = None
    patience_counter = 0

    if training_config.verbose:
        print(
            "Beginning GP training "
            f"(model={model_config.label!r}, "
            f"patience={training_config.patience}, "
            f"min_delta={training_config.min_delta})."
        )

    for iteration in range(training_config.training_iter):
        optimizer.zero_grad(set_to_none=True)

        output = model(train_x)
        loss = -marginal_log_likelihood(output, train_y)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite GP loss at iteration {iteration + 1}.")

        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu())

        if (
            training_config.verbose
            and (iteration + 1) % training_config.log_interval == 0
        ):
            print(
                f"Iter {iteration + 1}/{training_config.training_iter} "
                f"- Loss: {current_loss:.6f}"
            )

        if current_loss < best_loss - training_config.min_delta:
            best_loss = current_loss
            best_iteration = iteration
            best_model_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= training_config.patience:
            if training_config.verbose:
                print(
                    "Early stopping at iteration "
                    f"{iteration + 1}; restoring iteration "
                    f"{best_iteration + 1} with loss {best_loss:.6f}."
                )
            break

    if best_model_state is None or best_likelihood_state is None:
        raise RuntimeError("GP training did not produce a valid checkpoint.")

    # Restore the best checkpoint whether training stopped early or reached
    # the maximum iteration count.
    model.load_state_dict(best_model_state)
    likelihood.load_state_dict(best_likelihood_state)

    model.eval()
    likelihood.eval()

    if training_config.verbose:
        print(
            f"Restored best checkpoint from iteration {best_iteration + 1} "
            f"with loss {best_loss:.6f}."
        )

    return model, likelihood
