from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

import numpy as np
import torch


ARTIFACT_SCHEMA_VERSION = 2

SamplingMethod = Literal["random", "stratified"]
InitialCondition = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration of the reference physical simulator."""

    L: float = 10.0
    N: int = 60
    T: float = 5.5
    dt: float = 0.05
    stiffness_profile: str = "periodic"
    num_hinges: int = 4
    extra_kwargs: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.L <= 0.0:
            raise ValueError("L must be positive.")
        if self.N < 3:
            raise ValueError("N must be at least 3.")
        if self.T <= 0.0:
            raise ValueError("T must be positive.")
        if self.dt <= 0.0 or self.dt >= self.T:
            raise ValueError("dt must satisfy 0 < dt < T.")
        if self.num_hinges < 0:
            raise ValueError("num_hinges cannot be negative.")

    def simulator_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "L": self.L,
            "N": self.N,
            "T": self.T,
            "dt": self.dt,
            "stiffness_profile": self.stiffness_profile,
            "num_hinges": self.num_hinges,
        }
        kwargs.update(dict(self.extra_kwargs))
        return kwargs

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.L,
            self.N,
            self.T,
            self.dt,
            self.stiffness_profile,
            self.num_hinges,
            self.extra_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.simulator_kwargs()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration of one GP-PHS model variant."""

    label: str
    kernel_type: str = "baseline"
    strain_kernel: str = "rbf"
    spatial_kernel: str = "rbf"
    matern_nu: float = 2.5
    changepoint_location: float = 5.0
    changepoint_width: float = 0.15
    n_freqs: int = 10
    extra_params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("Model label cannot be empty.")
        if self.matern_nu <= 0.0:
            raise ValueError("matern_nu must be positive.")
        if self.changepoint_width <= 0.0:
            raise ValueError("changepoint_width must be positive.")
        if self.n_freqs <= 0:
            raise ValueError("n_freqs must be positive.")

    def training_params(self, simulator: SimulatorConfig) -> dict[str, Any]:
        """Build the parameter mapping expected by train_gp_model()."""
        params = {
            "stiffness_profile": simulator.stiffness_profile,
            "num_hinges": simulator.num_hinges,
            "strain_kernel": self.strain_kernel,
            "spatial_kernel": self.spatial_kernel,
            "matern_nu": self.matern_nu,
            "changepoint_location": self.changepoint_location,
            "changepoint_width": self.changepoint_width,
            "n_freqs": self.n_freqs,
        }
        params.update(dict(self.extra_params))
        return params

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kernel_type": self.kernel_type,
            "strain_kernel": self.strain_kernel,
            "spatial_kernel": self.spatial_kernel,
            "matern_nu": self.matern_nu,
            "changepoint_location": self.changepoint_location,
            "changepoint_width": self.changepoint_width,
            "n_freqs": self.n_freqs,
            **dict(self.extra_params),
        }


@dataclass(frozen=True)
class TrainingConfig:
    training_iter: int = 1000
    patience: int = 15
    normalization_eps: float = 1e-6
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.training_iter <= 0:
            raise ValueError("training_iter must be positive.")
        if self.patience <= 0:
            raise ValueError("patience must be positive.")
        if self.normalization_eps <= 0.0:
            raise ValueError("normalization_eps must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_iter": self.training_iter,
            "patience": self.patience,
            "normalization_eps": self.normalization_eps,
            "verbose": self.verbose,
        }


@dataclass(frozen=True)
class SamplingConfig:
    method: SamplingMethod = "stratified"
    train_fraction: float = 0.06
    left_region: tuple[float, float] = (0.0, 2.0)
    middle_region: tuple[float, float] = (2.0, 8.0)
    right_region: tuple[float, float] = (8.0, 10.0)
    weights: tuple[float, float, float] = (0.30, 0.40, 0.30)
    include_domain_boundaries: bool = True
    shuffle: bool = True
    z_col: int = 2

    def __post_init__(self) -> None:
        if self.method not in {"random", "stratified"}:
            raise ValueError(f"Unsupported sampling method: {self.method!r}.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1.")
        if len(self.weights) != 3:
            raise ValueError("weights must contain exactly three values.")
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("sampling weights cannot be negative.")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("sampling weights must sum to 1.")

        regions = (self.left_region, self.middle_region, self.right_region)
        if any(start >= end for start, end in regions):
            raise ValueError("Each sampling region must satisfy start < end.")
        if self.left_region[1] > self.middle_region[0]:
            raise ValueError("left_region and middle_region overlap.")
        if self.middle_region[1] > self.right_region[0]:
            raise ValueError("middle_region and right_region overlap.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "train_fraction": self.train_fraction,
            "left_region": self.left_region,
            "middle_region": self.middle_region,
            "right_region": self.right_region,
            "weights": self.weights,
            "include_domain_boundaries": self.include_domain_boundaries,
            "shuffle": self.shuffle,
            "z_col": self.z_col,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    curl_max_points: int = 2048
    curl_batch_size: int = 256
    force_edge: int = 5
    include_likelihood_noise: bool = True
    rollout_method: str = "RK45"

    def __post_init__(self) -> None:
        if self.curl_max_points <= 0:
            raise ValueError("curl_max_points must be positive.")
        if self.curl_batch_size <= 0:
            raise ValueError("curl_batch_size must be positive.")
        if self.force_edge < 1:
            raise ValueError("force_edge must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "curl_max_points": self.curl_max_points,
            "curl_batch_size": self.curl_batch_size,
            "force_edge": self.force_edge,
            "include_likelihood_noise": self.include_likelihood_noise,
            "rollout_method": self.rollout_method,
        }


@dataclass(frozen=True)
class PipelineConfig:
    simulator: SimulatorConfig
    model: ModelConfig
    sampling: SamplingConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration for parallel execution across random seeds."""

    enabled: bool = True
    max_workers: int | None = None
    torch_threads_per_worker: int | None = None
    start_method: Literal["spawn", "forkserver"] = "spawn"

    def __post_init__(self) -> None:
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError(
                "max_workers must be positive or None."
            )

        if (
            self.torch_threads_per_worker is not None
            and self.torch_threads_per_worker <= 0
        ):
            raise ValueError(
                "torch_threads_per_worker must be positive or None."
            )


@dataclass(frozen=True)
class ExperimentSpec:
    """One physical profile and the model variants evaluated on it."""

    name: str
    row_label: str
    simulator: SimulatorConfig
    sampling: SamplingConfig
    models: tuple[ModelConfig, ...]
    seeds: tuple[int, ...] = (6, 42, 79, 123, 456)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty.")
        if not self.models:
            raise ValueError("Each experiment must contain at least one model.")
        if not self.seeds:
            raise ValueError("Each experiment must contain at least one seed.")

        labels = [model.label for model in self.models]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate model labels in experiment {self.name!r}.")


@dataclass
class NormalizationStats:
    X_mean: torch.Tensor
    X_std: torch.Tensor
    y_p_mean: torch.Tensor
    y_p_std: torch.Tensor
    y_q_mean: torch.Tensor
    y_q_std: torch.Tensor
    eps: float


@dataclass
class DatasetBundle:
    reference_data: Mapping[str, np.ndarray]
    X_all: np.ndarray
    y_p_all: np.ndarray
    y_q_all: np.ndarray
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    X_train_raw: torch.Tensor
    y_p_train_raw: torch.Tensor
    y_q_train_raw: torch.Tensor
    X_train_norm: torch.Tensor
    y_p_train_norm: torch.Tensor
    y_q_train_norm: torch.Tensor
    normalization: NormalizationStats
    grid_shape: tuple[int, int]


@dataclass
class SimulationBundle:
    simulator: Any
    training_data: Mapping[str, np.ndarray]
    evaluation_data: Mapping[str, np.ndarray]


@dataclass
class TrainedModels:
    dEdp_model: Any
    dEdp_likelihood: Any
    dEdq_model: Any
    dEdq_likelihood: Any


@dataclass
class RunResult:
    """Serializable result returned by one seed worker."""

    experiment_name: str
    model_label: str
    seed: int
    pipeline_config: PipelineConfig

    metrics: dict[str, Any]

    p_gp: np.ndarray
    q_gp: np.ndarray

    H_reference: np.ndarray
    H_gp: np.ndarray

    dHdq_summary: dict[str, np.ndarray]


@dataclass
class AggregateResult:
    experiment_name: str
    model_label: str
    runs: tuple[RunResult, ...]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    p_gp_mean: np.ndarray
    p_gp_std: np.ndarray
    q_gp_mean: np.ndarray
    q_gp_std: np.ndarray


@dataclass(frozen=True)
class SeedJob:
    """Everything needed to execute one seed in a worker process."""

    pipeline_config: PipelineConfig
    seed: int
    training_data: dict[str, np.ndarray]
    evaluation_data: dict[str, np.ndarray]