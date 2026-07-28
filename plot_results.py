from plotting import (
    load_experiment_artifact,
    plot_multi_profile_rollout_grid_from_artifact,
    plot_profile_overlay_from_artifact,
)


def plot_main_experiment(
    data_path: str = "./data/paper_experiment.pkl",
) -> None:
    artifact = load_experiment_artifact(data_path)

    plot_multi_profile_rollout_grid_from_artifact(
        artifact,
        time_slices=(0, 50, 100),
        save_path="./plots/full_rollout_test.pdf",
    )

    for experiment_name in artifact["experiment_order"]:
        plot_profile_overlay_from_artifact(
            artifact,
            experiment_name=experiment_name,
            time_slices=(0, 50, 100),
            save_path=f"./plots/{experiment_name}_rollout_overlay.pdf",
        )


if __name__ == "__main__":
    plot_main_experiment(data_path="./data/paper_experiment.pkl")
