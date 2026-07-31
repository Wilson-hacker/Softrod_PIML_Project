import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config import (
    ARTIFACT_SCHEMA_VERSION,
    AggregateResult,
)
from plotting import (
    load_experiment_artifact,
    plot_multi_profile_rollout_grid_from_artifact,
)


def plot_main_experiment(
    data_path: str = "./data/paper_experiment.pkl",
) -> None:
    artifact = load_experiment_artifact(data_path)

    plot_multi_profile_rollout_grid_from_artifact(
        artifact,
        experiment_filter=["periodic", "C2", "piecewise"],
        time_slices=(0, 50, 100),
        save_path="./plots/rollout_consistent_scaling.pdf",
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
                f"{aggregate.mean_metrics[column]:>9.3e}±"
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


if __name__ == "__main__":
    results = load_aggregate_results_from_artifact("./data/paper_experiment_new.pkl")
    print_aggregate_table(results)
    plot_main_experiment(data_path="./data/paper_experiment_new.pkl")
