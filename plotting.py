from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import ARTIFACT_SCHEMA_VERSION


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


def plot_multi_profile_rollout_grid_from_artifact(
    artifact: Mapping[str, Any],
    experiment_filter: Sequence[str] | None = None,
    time_slices: Sequence[int] = (0, 50, 100),
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
) -> None:
    experiments = artifact["experiments"]

    if experiment_filter is None:
        experiment_order = artifact["experiment_order"]
    else:
        missing = set(experiment_filter) - set(experiments.keys())

        if missing:
            raise KeyError(f"Unknown experiments requested: {sorted(missing)}")

        experiment_order = [
            name for name in artifact["experiment_order"] if name in experiment_filter
        ]

    if not experiment_order:
        raise ValueError("No experiments selected for plotting.")

    figure, axes = plt.subplots(
        len(experiment_order),
        len(time_slices),
        figsize=figsize,
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    legend_handles = None
    legend_labels = None

    for row, experiment_name in enumerate(experiment_order):
        profile = experiments[experiment_name]

        z = np.asarray(profile["z"])
        dt = float(profile["dt"])
        dz = float(profile["dz"])

        u_reference = np.asarray(profile["u_reference"])

        model_displacements = {}

        for model_label, model in profile["models"].items():
            model_displacements[model_label] = np.asarray(model["u_gp_mean"])

        for column, step in enumerate(time_slices):
            if not 0 <= step < u_reference.shape[1]:
                raise IndexError(
                    f"Invalid time index {step} for experiment {experiment_name!r}."
                )

            axis = axes[row, column]
            (reference_line,) = axis.plot(
                z,
                u_reference[:, step],
                label="Numerical solution",
            )

            model_lines = []
            for model_label, u_prediction in model_displacements.items():
                (line,) = axis.plot(
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
                legend_labels = ["Numerical solution", *model_displacements.keys()]

    if legend_handles and legend_labels:
        figure.legend(
            legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels), frameon=True
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


def plot_force_field_error_heatmap_from_artifact(
    artifact: Mapping[str, Any],
    experiment_name: str,
    model_label: str,
    log_scale: bool = False,
    edge: int = 5,
    save_path: str | Path | None = None,
) -> None:

    profile = artifact["experiments"][experiment_name]
    model = profile["models"][model_label]
    force_error = np.asarray(model["dHdq_force_error"])

    if force_error.ndim == 1:
        N_z, N_t = profile["grid_shape"]

        force_error = force_error.reshape(N_z, N_t)

    force_error_abs = np.abs(force_error)
    plot_error = force_error_abs.copy()

    plot_error[:edge, :] = np.nan
    plot_error[-edge:, :] = np.nan

    if log_scale:
        plot_error = np.log10(plot_error + 1e-12)

        cbar_label = (
            r"$\log_{10}"
            r"(|\Delta \partial_z(\partial_q H)|"
            r"+10^{-12})$"
        )

    else:
        cbar_label = r"$|\Delta \partial_z(\partial_q H)|$"

    z = np.asarray(profile["z"])
    dt = float(profile["dt"])
    N_t = force_error.shape[1]
    t = np.arange(N_t) * dt

    figure, axis = plt.subplots(figsize=(9, 4.8))
    image = axis.imshow(
        plot_error,
        origin="lower",
        aspect="auto",
        extent=(t[0], t[-1], z[0], z[-1]),
    )

    figure.colorbar(image, ax=axis, label=cbar_label)

    axis.set_xlabel("t")
    axis.set_ylabel("z")

    axis.set_title(f"{experiment_name}: {model_label}")

    figure.tight_layout()
    _finalize_figure(figure, save_path)
