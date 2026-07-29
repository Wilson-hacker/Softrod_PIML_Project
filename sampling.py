from __future__ import annotations

import random
from typing import Mapping

import numpy as np
import torch

from config import DatasetBundle, NormalizationStats, SamplingConfig


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible sequential runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_z_indices(
    X: np.ndarray,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    num_train = int(X.shape[0] * train_fraction)
    if num_train <= 0:
        raise ValueError("The requested training fraction selects no samples.")

    rng = np.random.default_rng(seed)
    return rng.choice(X.shape[0], size=num_train, replace=False)


def stratified_z_indices(
    X: np.ndarray,
    train_fraction: float,
    seed: int = 42,
    z_col: int = 2,
    left_region: tuple[float, float] = (0.0, 2.0),
    middle_region: tuple[float, float] = (2.0, 8.0),
    right_region: tuple[float, float] = (8.0, 10.0),
    weights: tuple[float, float, float] = (0.30, 0.40, 0.30),
    include_domain_boundaries: bool = True,
    shuffle: bool = True,
) -> np.ndarray:
    """Return spatially stratified indices selected without replacement."""
    rng = np.random.default_rng(seed)
    z = X[:, z_col]
    num_total = X.shape[0]
    num_train = int(num_total * train_fraction)

    if num_train <= 0:
        raise ValueError("The requested training fraction selects no samples.")

    n_left = int(round(num_train * weights[0]))
    n_middle = int(round(num_train * weights[1]))
    n_right = num_train - n_left - n_middle

    if include_domain_boundaries:
        masks = (
            (z >= left_region[0]) & (z < left_region[1]),
            (z >= middle_region[0]) & (z < middle_region[1]),
            (z >= right_region[0]) & (z <= right_region[1]),
        )
    else:
        masks = (
            (z > left_region[0]) & (z < left_region[1]),
            (z > middle_region[0]) & (z < middle_region[1]),
            (z > right_region[0]) & (z < right_region[1]),
        )

    pools = tuple(np.flatnonzero(mask) for mask in masks)
    requested = (n_left, n_middle, n_right)
    selected_parts: list[np.ndarray] = []

    for pool, count in zip(pools, requested):
        actual_count = min(count, pool.size)
        if actual_count:
            selected_parts.append(rng.choice(pool, size=actual_count, replace=False))

    train_indices = (
        np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=int)
    )

    if train_indices.size < num_train:
        selected_mask = np.zeros(num_total, dtype=bool)
        selected_mask[train_indices] = True
        remaining_pool = np.flatnonzero(~selected_mask)
        missing = num_train - train_indices.size

        if remaining_pool.size < missing:
            raise ValueError("Not enough observations to fill the training budget.")

        extra = rng.choice(remaining_pool, size=missing, replace=False)
        train_indices = np.concatenate((train_indices, extra))

    if shuffle:
        rng.shuffle(train_indices)

    return train_indices


def select_training_indices(
    X: np.ndarray,
    config: SamplingConfig,
    seed: int,
) -> np.ndarray:
    if config.method == "random":
        return random_z_indices(X, config.train_fraction, seed)

    return stratified_z_indices(
        X=X,
        train_fraction=config.train_fraction,
        seed=seed,
        z_col=config.z_col,
        left_region=config.left_region,
        middle_region=config.middle_region,
        right_region=config.right_region,
        weights=config.weights,
        include_domain_boundaries=config.include_domain_boundaries,
        shuffle=config.shuffle,
    )


def make_flat_dataset_from_sim_data(
    data: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct X=[p,q,z], dH/dp targets, and dH/dq targets."""
    p = np.asarray(data["p"])
    q = np.asarray(data["q"])
    z = np.asarray(data["z"])

    if p.shape != q.shape:
        raise ValueError("p and q trajectories must have the same shape.")
    if p.shape[0] != z.size:
        raise ValueError("The first trajectory dimension must match len(z).")

    n_z, n_t = q.shape
    z_grid = np.broadcast_to(z[:, None], (n_z, n_t))

    X_all = np.column_stack((p.reshape(-1), q.reshape(-1), z_grid.reshape(-1)))
    y_p_all = np.asarray(data["dH_dp"]).reshape(-1)
    y_q_all = np.asarray(data["dH_dq"]).reshape(-1)

    if not (X_all.shape[0] == y_p_all.size == y_q_all.size):
        raise ValueError("Flattened inputs and targets have inconsistent sizes.")

    return X_all, y_p_all, y_q_all


def prepare_training_dataset(
    reference_data: Mapping[str, np.ndarray],
    sampling_config: SamplingConfig,
    seed: int,
    normalization_eps: float = 1e-6,
) -> DatasetBundle:
    """Create training tensors, exact held-out indices, and normalization data."""
    X_all, y_p_all, y_q_all = make_flat_dataset_from_sim_data(reference_data)
    train_indices = select_training_indices(X_all, sampling_config, seed)

    heldout_mask = np.ones(X_all.shape[0], dtype=bool)
    heldout_mask[train_indices] = False
    heldout_indices = np.flatnonzero(heldout_mask)

    X_train_raw = torch.as_tensor(X_all[train_indices], dtype=torch.float64)
    y_p_train_raw = torch.as_tensor(y_p_all[train_indices], dtype=torch.float64)
    y_q_train_raw = torch.as_tensor(y_q_all[train_indices], dtype=torch.float64)

    X_mean = X_train_raw.mean(dim=0)
    X_std = X_train_raw.std(dim=0)
    y_p_mean = y_p_train_raw.mean()
    y_p_std = y_p_train_raw.std()
    y_q_mean = y_q_train_raw.mean()
    y_q_std = y_q_train_raw.std()

    normalization = NormalizationStats(
        X_mean=X_mean,
        X_std=X_std,
        y_p_mean=y_p_mean,
        y_p_std=y_p_std,
        y_q_mean=y_q_mean,
        y_q_std=y_q_std,
        eps=normalization_eps,
    )

    X_train_norm = (X_train_raw - X_mean) / (X_std + normalization_eps)
    y_p_train_norm = (y_p_train_raw - y_p_mean) / (y_p_std + normalization_eps)
    y_q_train_norm = (y_q_train_raw - y_q_mean) / (y_q_std + normalization_eps)

    grid_shape = tuple(int(value) for value in np.asarray(reference_data["q"]).shape)

    return DatasetBundle(
        reference_data=reference_data,
        X_all=X_all,
        y_p_all=y_p_all,
        y_q_all=y_q_all,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        X_train_raw=X_train_raw,
        y_p_train_raw=y_p_train_raw,
        y_q_train_raw=y_q_train_raw,
        X_train_norm=X_train_norm,
        y_p_train_norm=y_p_train_norm,
        y_q_train_norm=y_q_train_norm,
        normalization=normalization,
        grid_shape=grid_shape,
    )


def normalize_changepoint_params(
    location_physical: float,
    width_physical: float,
    normalization: NormalizationStats,
    z_col: int = 2,
) -> tuple[float, float]:
    z_mean = float(normalization.X_mean[z_col].detach().cpu())
    z_std = float(normalization.X_std[z_col].detach().cpu()) + normalization.eps
    return (location_physical - z_mean) / z_std, width_physical / z_std
