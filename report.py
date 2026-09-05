import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from config import (
    ARTIFACT_SCHEMA_VERSION,
    AggregateResult,
)
from plotting import (
    create_rollout_video_from_artifact,
    load_experiment_artifact,
    plot_multi_profile_rollout_grid_from_artifact,
)


def plot_main_experiment(
    data_path: str = "./data/paper_experiment.pkl",
    save_path: str = "./plots/rollout_plts.pdf",
) -> None:
    artifact = load_experiment_artifact(data_path)

    plot_multi_profile_rollout_grid_from_artifact(
        artifact,
        experiment_filter=["periodic", "C2", "piecewise"],
        time_slices=(0, 50, 100),
        save_path=save_path,
    )


def print_aggregate_table(
    results: Mapping[str, Mapping[str, AggregateResult]],
) -> None:
    columns = (
        "q_rel_l2",
        "u_rel_l2",
        "energy_rel_l2",
        "curl_rms",
        "dHdq_rmse",
        "dHdq_force_rmse",
    )
    header = f"{'Experiment':<16} {'Model':<14} " + " ".join(
        f"{column:>17}" for column in columns
    )
    print("\n" + header)
    print("-" * len(header))

    for experiment_name, model_results in results.items():
        for model_label, aggregate in model_results.items():
            values = " ".join(
                f"{aggregate.mean_metrics[column]:>9.2e}±"
                f"{aggregate.std_metrics[column]:<7.1e}"
                for column in columns
            )
            print(f"{experiment_name:<16} {model_label:<14} {values}")


def load_aggregate_results_from_artifact(
    artifact_path: str | Path,
) -> dict[str, dict[str, AggregateResult]]:
    """
    Load a saved experiment artifact and reconstruct a results dictionary
    compatible with print_aggregate_table().

    Only load pickle files that you created or trust, because pickle files
    can execute arbitrary code during deserialization.

    Parameters
    ----------
    artifact_path:
        Path to the .pkl artifact produced by run_experiment_matrix().

    Returns
    -------
    dict[str, dict[str, AggregateResult]]
        Nested mapping:

            results[experiment_name][model_label] = AggregateResult(...)

        The reconstructed AggregateResult objects contain the stored aggregate
        metrics and rollout statistics. Their ``runs`` field is empty because
        individual RunResult objects are not serialized in the artifact.
    """
    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Experiment artifact not found: '{artifact_path}'.")

    with artifact_path.open("rb") as file:
        artifact: dict[str, Any] = pickle.load(file)

    schema_version = artifact.get("schema_version")

    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported artifact schema version: "
            f"{schema_version!r}. Expected "
            f"{ARTIFACT_SCHEMA_VERSION}."
        )

    experiments = artifact.get("experiments")

    if not isinstance(experiments, dict):
        raise TypeError(
            "The artifact does not contain a valid 'experiments' dictionary."
        )

    experiment_order = artifact.get(
        "experiment_order",
        tuple(experiments.keys()),
    )

    results: dict[
        str,
        dict[str, AggregateResult],
    ] = {}

    for experiment_name in experiment_order:
        if experiment_name not in experiments:
            raise ValueError(
                f"Experiment '{experiment_name}' is listed in "
                "'experiment_order' but is absent from 'experiments'."
            )

        experiment_data = experiments[experiment_name]

        models = experiment_data.get("models")

        if not isinstance(models, dict):
            raise TypeError(
                f"Experiment '{experiment_name}' does not contain "
                "a valid 'models' dictionary."
            )

        model_results: dict[
            str,
            AggregateResult,
        ] = {}

        for model_label, model_data in models.items():
            mean_metrics = model_data.get("mean_metrics")
            std_metrics = model_data.get("std_metrics")

            if not isinstance(
                mean_metrics,
                dict,
            ):
                raise TypeError(
                    f"Model '{model_label}' in experiment "
                    f"'{experiment_name}' has no valid "
                    "'mean_metrics' dictionary."
                )

            if not isinstance(
                std_metrics,
                dict,
            ):
                raise TypeError(
                    f"Model '{model_label}' in experiment "
                    f"'{experiment_name}' has no valid "
                    "'std_metrics' dictionary."
                )

            p_gp_mean = np.asarray(
                model_data.get(
                    "p_gp_mean",
                    np.empty((0, 0)),
                )
            )

            p_gp_std = np.asarray(
                model_data.get(
                    "p_gp_std",
                    np.empty_like(p_gp_mean),
                )
            )

            q_gp_mean = np.asarray(
                model_data.get(
                    "q_gp_mean",
                    np.empty((0, 0)),
                )
            )

            q_gp_std = np.asarray(
                model_data.get(
                    "q_gp_std",
                    np.empty_like(q_gp_mean),
                )
            )

            model_results[model_label] = AggregateResult(
                experiment_name=(experiment_name),
                model_label=model_label,
                runs=(),
                mean_metrics={key: float(value) for key, value in mean_metrics.items()},
                std_metrics={key: float(value) for key, value in std_metrics.items()},
                p_gp_mean=p_gp_mean,
                p_gp_std=p_gp_std,
                q_gp_mean=q_gp_mean,
                q_gp_std=q_gp_std,
            )

        results[experiment_name] = model_results

    return results


def check_right_boundary_momentum(
    artifact_path: str | Path,
    experiment_name: str,
    model_filter: Sequence[str] | None = None,
    tolerance: float = 1e-8,
) -> None:
    """
    Check whether p(L, t) = 0 for the reference solution and GP rollouts.

    Parameters
    ----------
    artifact_path:
        Path to the saved .pkl artifact.

    experiment_name:
        Experiment to inspect, e.g. "periodic", "C2", or "piecewise".

    model_filter:
        Optional model labels to check. If None, check all models.

    tolerance:
        Maximum allowed absolute boundary value.
    """
    with Path(artifact_path).open("rb") as file:
        artifact = pickle.load(file)

    experiments = artifact["experiments"]

    if experiment_name not in experiments:
        raise KeyError(
            f"Unknown experiment {experiment_name!r}."
        )

    profile = experiments[experiment_name]

    p_reference = np.asarray(
        profile["p_reference"],
        dtype=float,
    )

    if p_reference.ndim != 2:
        raise ValueError(
            "p_reference must have shape (N_z, N_t)."
        )

    p_reference_right = p_reference[-1, :]
    reference_max_error = np.max(
        np.abs(p_reference_right)
    )

    print(
        f"Experiment: {experiment_name}"
    )
    print(
        "Reference: "
        f"max |p(L,t)| = {reference_max_error:.3e} "
        f"-> {'PASS' if reference_max_error <= tolerance else 'FAIL'}"
    )

    models = profile["models"]

    if model_filter is None:
        model_labels = list(models.keys())
    else:
        missing = set(model_filter) - set(
            models.keys()
        )

        if missing:
            raise KeyError(
                f"Unknown model labels: {sorted(missing)}"
            )

        model_labels = list(model_filter)

    for model_label in model_labels:
        p_gp = np.asarray(
            models[model_label]["p_gp_mean"],
            dtype=float,
        )

        if p_gp.shape != p_reference.shape:
            raise ValueError(
                f"{model_label!r} p_gp_mean has shape "
                f"{p_gp.shape}, expected {p_reference.shape}."
            )

        p_gp_right = p_gp[-1, :]
        model_max_error = np.max(
            np.abs(p_gp_right)
        )

        print(
            f"{model_label}: "
            f"max |p(L,t)| = {model_max_error:.3e} "
            f"-> {'PASS' if model_max_error <= tolerance else 'FAIL'}"
        )


if __name__ == "__main__":
    result_path = "./data/paper_experiment_n100.pkl"
    results = load_aggregate_results_from_artifact(result_path)
    print_aggregate_table(results)
    # check_right_boundary_momentum(artifact_path=result_path, experiment_name="piecewise")
    create_rollout_video_from_artifact(
        artifact_path=result_path,
        experiment_name="piecewise",
        model_label="Baseline",
        save_path="./plots/piecewise_baseline2.mp4",
        frame_stride=1,
        fps=20,
    )
    # plot_main_experiment(data_path=result_path, save_path="./plots/rollout_n100.pdf")
