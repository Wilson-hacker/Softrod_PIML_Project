from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from config import ARTIFACT_SCHEMA_VERSION, ExperimentSpec
from simulator import build_stiffness_profile


def load_experiment_artifact(path: str | Path) -> dict[str, Any]:
    """Load a trusted experiment artifact produced by run_experiments.py."""
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
    """Display the figure or save it and close it."""
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
    """Plot saved displacement rollouts in a profile-by-time grid."""
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
    """Plot saved effort and force RMSE curves over time."""
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
    """Plot the saved absolute force error over space and time."""
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


def plot_stiffness_profiles(
    experiment_specs: Sequence[ExperimentSpec],
    experiment_filter: Sequence[str] | None = None,
    num_plot_points: int = 1000,
    save_path: str | Path = "./plots/stiffness_profiles.pdf",
) -> None:
    """Plot stiffness profiles on a dense grid independent of the simulation grid.

    Use experiment_filter to select profiles in experiment_specs order.
    Set num_plot_points to control the plotting resolution.
    """
    if num_plot_points < 2:
        raise ValueError("num_plot_points must be at least 2.")

    if experiment_filter is None:
        selected_specs = list(experiment_specs)
    else:
        requested_names = set(experiment_filter)
        available_names = {experiment.name for experiment in experiment_specs}
        missing_names = requested_names - available_names

        if missing_names:
            raise ValueError(
                "Unknown experiment names in experiment_filter: "
                f"{sorted(missing_names)}."
            )

        selected_specs = [
            experiment
            for experiment in experiment_specs
            if experiment.name in requested_names
        ]

    if not selected_specs:
        raise ValueError("No stiffness profiles were selected for plotting.")

    fig, axes = plt.subplots(
        nrows=len(selected_specs),
        ncols=1,
        figsize=(6.5, 2.0 * len(selected_specs)),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    for axis, experiment in zip(
        axes,
        selected_specs,
        strict=True,
    ):
        simulator_config = experiment.simulator
        z_plot = np.linspace(0.0, simulator_config.L, num_plot_points)
        stiffness = build_stiffness_profile(config=simulator_config, z_arr=z_plot)

        axis.plot(z_plot, stiffness, lw=2.0)

        axis.set_ylabel(r"Stiffness $k(z)$", fontsize=10)
        axis.set_title(experiment.row_label, fontsize=13)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"Spatial $z$", fontsize=10)
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def create_rollout_video_from_artifact(
    artifact_path: str | Path,
    experiment_name: str,
    model_label: str,
    save_path: str | Path,
    frame_stride: int = 1,
    fps: int = 20,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> None:
    """Save a rollout animation from a trusted experiment artifact.

    The panels show displacement trajectories and pointwise strain error.
    Select a profile and model with experiment_name and model_label.
    Use frame_stride to subsample time steps and fps to set playback speed.
    MP4 output requires FFmpeg; GIF output uses Pillow.
    """
    artifact_path = Path(artifact_path)
    save_path = Path(save_path)

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    with artifact_path.open("rb") as file:
        artifact: Mapping[str, Any] = pickle.load(file)

    experiments = artifact["experiments"]

    if experiment_name not in experiments:
        raise KeyError(
            f"Unknown experiment {experiment_name!r}. "
            f"Available experiments: {list(experiments.keys())}"
        )

    profile = experiments[experiment_name]

    models = profile["models"]
    if model_label not in models:
        raise KeyError(
            f"Unknown model {model_label!r} for experiment "
            f"{experiment_name!r}. "
            f"Available models: {list(models.keys())}"
        )

    z = np.asarray(profile["z"], dtype=float)
    t = np.asarray(profile["t"], dtype=float)

    u_reference = np.asarray(profile["u_reference"], dtype=float)
    u_prediction = np.asarray(models[model_label]["u_gp_mean"], dtype=float)

    q_reference = np.asarray(profile["q_reference"], dtype=float)
    q_prediction = np.asarray(models[model_label]["q_gp_mean"], dtype=float)

    if u_reference.shape != u_prediction.shape:
        raise ValueError(
            "Reference and predicted displacements must have "
            f"the same shape, but received "
            f"{u_reference.shape} and {u_prediction.shape}."
        )

    if q_reference.shape != q_prediction.shape:
        raise ValueError(
            "Reference and predicted strains must have "
            f"the same shape, but received "
            f"{q_reference.shape} and {q_prediction.shape}."
        )

    if q_reference.shape != u_reference.shape:
        raise ValueError(
            "Strain and displacement arrays must share the same space-time grid."
        )

    if u_reference.shape[0] != z.size:
        raise ValueError("Spatial dimension of displacement does not match z.")

    if u_reference.shape[1] != t.size:
        raise ValueError("Temporal dimension of displacement does not match t.")

    q_error = q_prediction - q_reference

    frame_indices = np.arange(0, t.size, frame_stride, dtype=int)

    # Keep axis limits fixed across frames.
    displacement_min = min(float(np.min(u_reference)), float(np.min(u_prediction)))
    displacement_max = max(float(np.max(u_reference)), float(np.max(u_prediction)))

    displacement_range = displacement_max - displacement_min
    displacement_margin = max(0.05 * displacement_range, 1e-8)

    max_abs_error = float(np.max(np.abs(q_error)))

    if max_abs_error == 0.0:
        max_abs_error = 1e-8

    figure, (rollout_axis, error_axis) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": (2.0, 1.0)}
    )

    figure_title = figure.suptitle("", fontsize="large", y=0.97)

    # Plot displacement trajectories.
    (reference_line,) = rollout_axis.plot(
        z, u_reference[:, 0], label="Numerical solution"
    )

    (prediction_line,) = rollout_axis.plot(
        z, u_prediction[:, 0], linestyle="--", label=model_label
    )

    rollout_axis.set_ylabel(r"$u(z,t)$")
    rollout_axis.set_ylim(
        displacement_min - displacement_margin, displacement_max + displacement_margin
    )

    rollout_axis.grid(True, linestyle=":", alpha=0.45)

    rollout_axis.legend()

    # Plot strain error.
    (q_error_line,) = error_axis.plot(z, q_error[:, 0])

    error_axis.axhline(0.0, linewidth=1.0, linestyle=":")

    error_axis.set_xlabel(r"$z$")
    error_axis.set_ylabel(r"$\hat{q}-q$")

    error_axis.set_ylim(-1.05 * max_abs_error, 1.05 * max_abs_error)

    error_axis.grid(True, linestyle=":", alpha=0.45)

    error_title = error_axis.set_title("", fontsize="medium")

    def update(frame_number: int):
        """Update trajectories and error labels for one animation frame."""
        step = frame_indices[frame_number]

        # Update displacement trajectories.
        reference_u = u_reference[:, step]
        prediction_u = u_prediction[:, step]

        reference_line.set_ydata(reference_u)

        prediction_line.set_ydata(prediction_u)

        # Update strain error.
        reference_q = q_reference[:, step]
        prediction_q = q_prediction[:, step]

        strain_error = prediction_q - reference_q

        q_error_line.set_ydata(strain_error)

        # Update the frame title.
        figure_title.set_text(
            f"{profile['row_label']} — "
            f"{model_label} — "
            rf"$t={float(t[step]):.2f}$"
        )

        # Compute the relative strain error for this frame.
        q_reference_norm = np.linalg.norm(reference_q)

        if q_reference_norm > 1e-12:
            q_relative_l2_error = np.linalg.norm(strain_error) / q_reference_norm

            error_title.set_text(
                r"Strain prediction error: "
                rf"$\varepsilon_q(t)"
                rf"={float(q_relative_l2_error):.3e}$"
            )
        else:
            error_title.set_text(
                r"Strain prediction error: "
                r"$\varepsilon_q(t)$ undefined"
            )

        return (
            reference_line,
            prediction_line,
            q_error_line,
            figure_title,
            error_title,
        )

    animation = FuncAnimation(
        figure, update, frames=len(frame_indices), interval=1000 / fps, blit=False
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.10,
        top=0.90,
        hspace=0.32,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = save_path.suffix.lower()

    if suffix == ".mp4":
        writer = FFMpegWriter(fps=fps, bitrate=3000)
    elif suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        raise ValueError("save_path must end in .mp4 or .gif.")

    animation.save(save_path, writer=writer, dpi=150)

    plt.close(figure)
