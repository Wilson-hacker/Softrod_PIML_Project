from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import ARTIFACT_SCHEMA_VERSION
from rollout import reconstruct_displacement_from_strain


def load_experiment_artifact(path: str | Path) -> dict[str, Any]:
    """Load a trusted experiment artifact produced by main.py."""
    path = Path(path)
    with path.open("rb") as file:
        artifact = pickle.load(file)

    version = artifact.get("schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema version {version!r}; "
            f"expected {ARTIFACT_SCHEMA_VERSION}."
        )
    return artifact


def _finalize_figure(figure: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        plt.show()
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_rollout_time_slices_from_artifact(
    artifact: Mapping[str, Any],
    experiment_name: str,
    model_label: str,
    time_slices: Sequence[int] = (0, 50, 100),
    save_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    profile = artifact["experiments"][experiment_name]
    model = profile["models"][model_label]
    z = np.asarray(profile["z"])
    dt = float(profile["dt"])
    u_reference = np.asarray(profile["u_reference"])
    u_prediction = reconstruct_displacement_from_strain(
        np.asarray(model["q_gp_mean"]),
        float(profile["dz"]),
    )

    figure, axes = plt.subplots(
        1,
        len(time_slices),
        figsize=(5.0 * len(time_slices), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    if title is None:
        title = f"{profile['row_label']}: {model_label}"
    if title:
        figure.suptitle(title)

    for axis_index, step in enumerate(time_slices):
        if not 0 <= step < u_reference.shape[1]:
            raise IndexError(f"Time step {step} is out of bounds.")

        axis = axes[axis_index]
        axis.plot(z, u_reference[:, step], label="Numerical solution")
        axis.plot(
            z,
            u_prediction[:, step],
            linestyle="--",
            label=model_label,
        )
        axis.set_title(f"Step {step} ($t={step * dt:.2f}$)")
        axis.set_xlabel(r"$z$")
        axis.grid(True, linestyle=":", alpha=0.5)
        if axis_index == 0:
            axis.set_ylabel(r"$u(z,t)$")
        axis.legend()

    figure.tight_layout()
    _finalize_figure(figure, save_path)


def plot_profile_overlay_from_artifact(
    artifact: Mapping[str, Any],
    experiment_name: str,
    time_slices: Sequence[int] = (0, 50, 100),
    save_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    profile = artifact["experiments"][experiment_name]
    z = np.asarray(profile["z"])
    dt = float(profile["dt"])
    dz = float(profile["dz"])
    u_reference = np.asarray(profile["u_reference"])

    figure, axes = plt.subplots(
        1,
        len(time_slices),
        figsize=(5.0 * len(time_slices), 4.6),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    if title is None:
        title = f"Average rollout: {profile['row_label']}"
    if title:
        figure.suptitle(title)

    handles: list[Any] = []
    labels: list[str] = []

    for axis_index, step in enumerate(time_slices):
        if not 0 <= step < u_reference.shape[1]:
            raise IndexError(f"Time step {step} is out of bounds.")

        axis = axes[axis_index]
        reference_line, = axis.plot(
            z,
            u_reference[:, step],
            label="Numerical solution",
        )
        if axis_index == 0:
            handles.append(reference_line)
            labels.append("Numerical solution")

        for model_label, model in profile["models"].items():
            u_prediction = reconstruct_displacement_from_strain(
                np.asarray(model["q_gp_mean"]),
                dz,
            )
            line, = axis.plot(
                z,
                u_prediction[:, step],
                linestyle="--",
                label=model_label,
            )
            if axis_index == 0:
                handles.append(line)
                labels.append(model_label)

        axis.set_title(f"Step {step} ($t={step * dt:.2f}$)")
        axis.set_xlabel(r"$z$")
        axis.grid(True, linestyle=":", alpha=0.5)
        if axis_index == 0:
            axis.set_ylabel(r"$u(z,t)$")

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    _finalize_figure(figure, save_path)


def plot_multi_profile_rollout_grid_from_artifact(
    artifact: Mapping[str, Any],
    time_slices: Sequence[int] = (0, 50, 100),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> None:
    experiment_order = artifact["experiment_order"]
    experiments = artifact["experiments"]
    figure, axes = plt.subplots(
        len(experiment_order),
        len(time_slices),
        figsize=figsize,
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    legend_handles: list[Any] | None = None
    legend_labels: list[str] | None = None

    for row, experiment_name in enumerate(experiment_order):
        profile = experiments[experiment_name]
        z = np.asarray(profile["z"])
        dt = float(profile["dt"])
        dz = float(profile["dz"])
        u_reference = np.asarray(profile["u_reference"])

        for column, step in enumerate(time_slices):
            if not 0 <= step < u_reference.shape[1]:
                raise IndexError(
                    f"Time step {step} is invalid for experiment {experiment_name!r}."
                )

            axis = axes[row, column]
            reference_line, = axis.plot(
                z,
                u_reference[:, step],
                label="Numerical solution",
            )
            model_lines = []

            for model_label, model in profile["models"].items():
                u_prediction = reconstruct_displacement_from_strain(
                    np.asarray(model["q_gp_mean"]),
                    dz,
                )
                line, = axis.plot(
                    z,
                    u_prediction[:, step],
                    linestyle="--",
                    label=model_label,
                )
                model_lines.append(line)

            if row == 0:
                axis.set_title(f"Step {step} ($t={step * dt:.2f}$)")
            if row == len(experiment_order) - 1:
                axis.set_xlabel(r"$z$")
            if column == 0:
                axis.set_ylabel(r"$u(z,t)$")
                axis.annotate(
                    profile["row_label"],
                    xy=(-0.22, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

            axis.grid(True, linestyle=":", alpha=0.45)

            if row == 0 and column == 0:
                legend_handles = [reference_line, *model_lines]
                legend_labels = ["Numerical solution", *profile["models"].keys()]

    if legend_handles is not None and legend_labels is not None:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=True,
        )

    figure.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.93,
        bottom=0.13,
        wspace=0.10,
        hspace=0.16,
    )
    _finalize_figure(figure, save_path)


def plot_effort_force_curves_from_artifact(
    artifact: Mapping[str, Any],
    experiment_name: str,
    model_label: str,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    profile = artifact["experiments"][experiment_name]
    model = profile["models"][model_label]
    curves = model.get("mean_diagnostic_curves")
    if curves is None:
        raise KeyError(
            "The artifact does not contain mean_diagnostic_curves for this model."
        )

    t = np.asarray(curves["t"])
    figure, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    axes[0].plot(t, curves["dHdq_rmse_t"])
    axes[0].set_ylabel(r"RMSE of $\partial_q H$")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, curves["force_rmse_t"])
    axes[1].set_ylabel(r"RMSE of $\partial_z(\partial_q H)$")
    axes[1].set_xlabel(r"$t$")
    axes[1].grid(True, alpha=0.3)

    if title is None:
        title = f"{profile['row_label']}: {model_label}"
    if title:
        figure.suptitle(title)

    figure.tight_layout()
    _finalize_figure(figure, save_path)
