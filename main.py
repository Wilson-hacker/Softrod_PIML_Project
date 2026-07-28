from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    as_completed,
)
from tqdm.auto import tqdm
import torch
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    ARTIFACT_SCHEMA_VERSION,
    AggregateResult,
    DatasetBundle,
    EvaluationConfig,
    ExperimentSpec,
    InitialCondition,
    ModelConfig,
    PipelineConfig,
    RunResult,
    SamplingConfig,
    SimulationBundle,
    SimulatorConfig,
    TrainedModels,
    TrainingConfig,
    ParallelConfig,
    SeedJob,
)
from metrics import (
    aggregate_run_results,
    evaluate_gp_phs_model,
    extract_selected_metrics,
    print_metrics_report,
)
from model import train_gp_model
from rollout import (
    attach_normalization_stats,
    reconstruct_displacement_from_strain,
    simulate_gp_phs,
)
from sampling import (
    normalize_changepoint_params,
    prepare_training_dataset,
    seed_everything,
)
from simulator import SoftRodSimulator


def default_initial_q(z: np.ndarray, length: float) -> np.ndarray:
    """Initial strain profile used in the paper experiments."""
    center = length / 2.0
    return -4.0 * (z - center) * np.exp(-(z - center) ** 2)


def initialize_seed_worker(
        torch_threads: int,
) -> None:
    """
    Configure CPU threading inside one worker process.

    Each process trains an independent model, so unrestricted Torch
    threading can otherwise oversubscribe the available CPU cores.
    """
    torch.set_num_threads(torch_threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts.
        pass


def resolve_parallel_resources(
        parallel_config: ParallelConfig,
        number_of_jobs: int,
) -> tuple[int, int]:
    """
    Determine the number of processes and Torch threads per process.
    """
    available_cpus = os.cpu_count() or 1

    if parallel_config.max_workers is None:
        max_workers = min(
            number_of_jobs,
            available_cpus,
        )
    else:
        max_workers = min(
            number_of_jobs,
            parallel_config.max_workers,
            available_cpus,
        )

    max_workers = max(1, max_workers)

    if parallel_config.torch_threads_per_worker is None:
        torch_threads = max(
            1,
            available_cpus // max_workers,
        )
    else:
        torch_threads = (
            parallel_config.torch_threads_per_worker
        )

    return max_workers, torch_threads


def create_simulation_bundle(
        config: SimulatorConfig,
        training_initial_condition: InitialCondition = default_initial_q,
        evaluation_initial_condition: InitialCondition | None = None,
) -> SimulationBundle:
    """Generate deterministic reference trajectories once per simulator setup."""
    simulator = SoftRodSimulator(**config.simulator_kwargs())
    training_data = simulator.generate_dataset(training_initial_condition)
    evaluation_data = (
        training_data
        if evaluation_initial_condition is None
        else simulator.generate_dataset(evaluation_initial_condition)
    )
    return SimulationBundle(
        simulator=simulator,
        training_data=training_data,
        evaluation_data=evaluation_data,
    )


def get_training_dataset(
        simulation: SimulationBundle,
        sampling_config: SamplingConfig,
        training_config: TrainingConfig,
        seed: int,
) -> DatasetBundle:
    """Prepare one seed's exact training set, held-out set, and targets."""
    return prepare_training_dataset(
        reference_data=simulation.training_data,
        sampling_config=sampling_config,
        seed=seed,
        normalization_eps=training_config.normalization_eps,
    )


def normalized_model_parameters(
        pipeline_config: PipelineConfig,
        dataset: DatasetBundle,
) -> dict[str, object]:
    params = pipeline_config.model.training_params(pipeline_config.simulator)
    if (
            pipeline_config.model.kernel_type == "SD"
            and pipeline_config.model.spatial_kernel == "changepoint"
    ):
        location, width = normalize_changepoint_params(
            location_physical=pipeline_config.model.changepoint_location,
            width_physical=pipeline_config.model.changepoint_width,
            normalization=dataset.normalization,
        )
        params["changepoint_location"] = location
        params["changepoint_width"] = width
    return params


def train_effort_models(
        pipeline_config: PipelineConfig,
        dataset: DatasetBundle,
) -> TrainedModels:
    """Train the momentum- and constitutive-effort GP models."""
    model_params = normalized_model_parameters(pipeline_config, dataset)
    training = pipeline_config.training

    dEdp_model, dEdp_likelihood = train_gp_model(
        dataset.X_train_norm,
        dataset.y_p_train_norm,
        model_params,
        kernel_type="baseline",
        training_iter=training.training_iter,
        patience=training.patience,
        verbose=training.verbose,
    )
    dEdq_model, dEdq_likelihood = train_gp_model(
        dataset.X_train_norm,
        dataset.y_q_train_norm,
        model_params,
        kernel_type=pipeline_config.model.kernel_type,
        training_iter=training.training_iter,
        patience=training.patience,
        verbose=training.verbose,
    )

    models = TrainedModels(
        dEdp_model=dEdp_model,
        dEdp_likelihood=dEdp_likelihood,
        dEdq_model=dEdq_model,
        dEdq_likelihood=dEdq_likelihood,
    )
    attach_normalization_stats(models, dataset.normalization)
    return models


def run_pipeline(
        pipeline_config: PipelineConfig,
        seed: int = 42,
        simulation: SimulationBundle | None = None,
        training_initial_condition: InitialCondition = default_initial_q,
        evaluation_initial_condition: InitialCondition | None = None,
) -> RunResult:
    """Compute one model/seed result. This function never creates plots."""
    seed_everything(seed)

    if simulation is None:
        simulation = create_simulation_bundle(
            config=pipeline_config.simulator,
            training_initial_condition=training_initial_condition,
            evaluation_initial_condition=evaluation_initial_condition,
        )
    verbose = pipeline_config.training.verbose

    if verbose:
        print("\n" + "=" * 80)
        print(
            f"Profile={pipeline_config.simulator.stiffness_profile} | "
            f"Model={pipeline_config.model.label} | Seed={seed}"
        )
        print("=" * 80)

        print("[1/4] Preparing the training dataset")
    dataset = get_training_dataset(
        simulation,
        pipeline_config.sampling,
        pipeline_config.training,
        seed,
    )

    if verbose:
        print("[2/4] Training the effort models")
    trained_models = train_effort_models(pipeline_config, dataset)

    if verbose:
        print("[3/4] Running the learned-model rollout")
    rollout_initial_condition = (
        training_initial_condition
        if evaluation_initial_condition is None
        else evaluation_initial_condition
    )
    p_gp, q_gp = simulate_gp_phs(
        simulator=simulation.simulator,
        models=trained_models,
        initial_condition=rollout_initial_condition,
        method=pipeline_config.evaluation.rollout_method,
    )

    if verbose:
        print("[4/4] Evaluating the model")
    metrics, dHdq_summary, H_reference, H_gp = evaluate_gp_phs_model(
        evaluation_data=simulation.evaluation_data,
        simulator=simulation.simulator,
        p_gp=p_gp,
        q_gp=q_gp,
        models=trained_models,
        dataset=dataset,
        evaluation_config=pipeline_config.evaluation,
        seed=seed,
    )

    if pipeline_config.training.verbose:
        print_metrics_report(metrics)

    return RunResult(
        experiment_name=(
            pipeline_config.simulator.stiffness_profile
        ),
        model_label=pipeline_config.model.label,
        seed=seed,
        pipeline_config=pipeline_config,
        metrics=metrics,
        p_gp=np.asarray(p_gp),
        q_gp=np.asarray(q_gp),
        H_reference=np.asarray(H_reference),
        H_gp=np.asarray(H_gp),
        dHdq_summary={
            key: np.asarray(value)
            for key, value in dHdq_summary.items()
        },
    )


def run_seed_worker(job: SeedJob) -> RunResult:
    """
    Execute one independent seed.

    The simulator object is reconstructed inside the process instead of
    being transferred between processes. The precomputed reference arrays
    are reused, so the reference PDE does not need to be solved again.
    """
    simulator = SoftRodSimulator(
        **job.pipeline_config.simulator.simulator_kwargs()
    )

    simulation = SimulationBundle(
        simulator=simulator,
        training_data=job.training_data,
        evaluation_data=job.evaluation_data,
    )

    return run_pipeline(
        pipeline_config=job.pipeline_config,
        seed=job.seed,
        simulation=simulation,
    )


def run_model_over_seeds(
    pipeline_config: PipelineConfig,
    seeds: Sequence[int],
    simulation: SimulationBundle | None = None,
    parallel_config: ParallelConfig | None = None,
) -> AggregateResult:
    """
    Compute one model configuration over several random seeds.

    Seed jobs are executed sequentially when parallel execution is disabled
    and in separate processes otherwise.
    """
    if not seeds:
        raise ValueError(
            "At least one seed must be provided."
        )

    if simulation is None:
        simulation = create_simulation_bundle(
            pipeline_config.simulator
        )

    parallel_config = (
        parallel_config
        or ParallelConfig(enabled=False)
    )

    progress_description = (
        f"{pipeline_config.model.label} seeds"
    )

    # Sequential execution.
    if (
        not parallel_config.enabled
        or len(seeds) == 1
    ):
        runs = tuple(
            run_pipeline(
                pipeline_config=pipeline_config,
                seed=seed,
                simulation=simulation,
            )
            for seed in tqdm(
                seeds,
                total=len(seeds),
                desc=progress_description,
                unit="seed",
            )
        )

        return aggregate_run_results(runs)

    max_workers, torch_threads = (
        resolve_parallel_resources(
            parallel_config,
            number_of_jobs=len(seeds),
        )
    )

    print(
        f"\nRunning {len(seeds)} seeds using "
        f"{max_workers} processes and "
        f"{torch_threads} Torch threads per process."
    )

    training_data = {
        key: np.asarray(value)
        for key, value in simulation.training_data.items()
    }

    evaluation_data = {
        key: np.asarray(value)
        for key, value in simulation.evaluation_data.items()
    }

    jobs = [
        SeedJob(
            pipeline_config=pipeline_config,
            seed=seed,
            training_data=training_data,
            evaluation_data=evaluation_data,
        )
        for seed in seeds
    ]

    context = mp.get_context(
        parallel_config.start_method
    )

    results_by_seed: dict[int, RunResult] = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=initialize_seed_worker,
        initargs=(torch_threads,),
    ) as executor:
        future_to_seed: dict[
            Future[RunResult],
            int,
        ] = {
            executor.submit(
                run_seed_worker,
                job,
            ): job.seed
            for job in jobs
        }

        current_seed: int | str = "unknown"

        try:
            completed_futures = as_completed(
                future_to_seed
            )

            for future in tqdm(
                completed_futures,
                total=len(future_to_seed),
                desc=progress_description,
                unit="seed",
            ):
                current_seed = future_to_seed[future]

                result = future.result()
                results_by_seed[current_seed] = result

                # Optional: display completed seeds without
                # corrupting the progress bar.
                tqdm.write(
                    f"[COMPLETED] "
                    f"{pipeline_config.model.label}, "
                    f"seed={current_seed}"
                )

        except Exception as error:
            for submitted_future in future_to_seed:
                submitted_future.cancel()

            raise RuntimeError(
                "Parallel run failed for seed "
                f"{current_seed}."
            ) from error

    # Restore the user-specified seed order.
    runs = tuple(
        results_by_seed[seed]
        for seed in seeds
    )

    return aggregate_run_results(runs)


def compute_mean_diagnostic_curves(
        aggregate: AggregateResult,
        edge: int,
) -> dict[str, np.ndarray]:
    """Average time-dependent effort and force errors over seeds."""
    dHdq_rmse_curves: list[np.ndarray] = []
    force_rmse_curves: list[np.ndarray] = []

    for run in aggregate.runs:
        summary = run.dHdq_summary
        n_z, n_t = run.q_gp.shape
        interior = slice(edge, n_z - edge)
        dHdq_pred = np.asarray(summary["dHdq_mean"]).reshape(n_z, n_t)
        dHdq_true = np.asarray(summary["dHdq_true"]).reshape(n_z, n_t)
        force_pred = np.asarray(summary["dHdq_force_pred"]).reshape(n_z, n_t)
        force_true = np.asarray(summary["dHdq_force_true"]).reshape(n_z, n_t)

        dHdq_rmse_curves.append(
            np.sqrt(np.nanmean((dHdq_pred[interior] - dHdq_true[interior]) ** 2, axis=0))
        )
        force_rmse_curves.append(
            np.sqrt(np.nanmean((force_pred[interior] - force_true[interior]) ** 2, axis=0))
        )

    dt = aggregate.runs[0].pipeline_config.simulator.dt
    return {
        "t": np.arange(aggregate.q_gp_mean.shape[1]) * dt,
        "dHdq_rmse_t": np.mean(np.stack(dHdq_rmse_curves), axis=0),
        "force_rmse_t": np.mean(np.stack(force_rmse_curves), axis=0),
    }


def aggregate_to_artifact_model(
        aggregate: AggregateResult,
        evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """Strip nonportable GP objects from an aggregate result."""
    return {
        "model_label": aggregate.model_label,
        "model_config": aggregate.runs[0].pipeline_config.model.to_dict(),
        "seeds": tuple(run.seed for run in aggregate.runs),
        "mean_metrics": dict(aggregate.mean_metrics),
        "std_metrics": dict(aggregate.std_metrics),
        "per_seed_metrics": {
            run.seed: extract_selected_metrics(run.metrics)
            for run in aggregate.runs
        },
        "p_gp_mean": np.asarray(aggregate.p_gp_mean),
        "p_gp_std": np.asarray(aggregate.p_gp_std),
        "q_gp_mean": np.asarray(aggregate.q_gp_mean),
        "q_gp_std": np.asarray(aggregate.q_gp_std),
        "mean_diagnostic_curves": compute_mean_diagnostic_curves(
            aggregate,
            evaluation_config.force_edge,
        ),
    }


def build_artifact_profile(
        experiment: ExperimentSpec,
        model_results: Mapping[str, AggregateResult],
        simulation: SimulationBundle,
        training_config: TrainingConfig,
        evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    simulator = simulation.simulator
    evaluation_data = simulation.evaluation_data

    q_reference = np.asarray(
        evaluation_data["q"]
    )

    return {
        "name": experiment.name,
        "row_label": experiment.row_label,
        "simulator_config": (
            experiment.simulator.to_dict()
        ),
        "sampling_config": (
            experiment.sampling.to_dict()
        ),
        "training_config": (
            training_config.to_dict()
        ),
        "evaluation_config": (
            evaluation_config.to_dict()
        ),
        "z": np.asarray(simulator.z),
        "dz": float(simulator.dz),
        "dt": float(simulator.dt),
        "t": np.asarray(
            evaluation_data.get(
                "t",
                np.arange(q_reference.shape[1])
                * simulator.dt,
            )
        ),
        "p_reference": np.asarray(
            evaluation_data["p"]
        ),
        "q_reference": q_reference,
        "u_reference": (
            reconstruct_displacement_from_strain(
                q_reference,
                simulator,
            )
        ),
        "models": {
            label: aggregate_to_artifact_model(
                result,
                evaluation_config,
            )
            for label, result
            in model_results.items()
        },
    }


def save_experiment_artifact(
        artifact: Mapping[str, Any],
        save_path: str | Path,
) -> None:
    """Atomically save a model-free computation artifact."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(dict(artifact), file, protocol=pickle.HIGHEST_PROTOCOL)

    temporary_path.replace(path)


def run_experiment_matrix(
        experiment_specs: Sequence[ExperimentSpec],
        save_results_path: str | Path,
        training_config: TrainingConfig | None = None,
        evaluation_config: EvaluationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
) -> dict[str, dict[str, AggregateResult]]:
    """
    Compute a profile/model/seed matrix and save plotting-ready data.

    The artifact is updated after every completed profile, so plotting can be
    performed later without loading GP models or rerunning training.
    """
    training_config = training_config or TrainingConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    parallel_config = (
            parallel_config
            or ParallelConfig(enabled=False)
    )
    results: dict[str, dict[str, AggregateResult]] = {}
    simulation_cache: dict[tuple[object, ...], SimulationBundle] = {}

    artifact: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_order": [],
        "experiments": {},
    }

    for experiment in experiment_specs:
        cache_key = experiment.simulator.cache_key()
        simulation = simulation_cache.get(cache_key)
        if simulation is None:
            print("\n" + "#" * 88)
            print(f"Generating reference data for {experiment.name}")
            print("#" * 88)
            simulation = create_simulation_bundle(experiment.simulator)
            simulation_cache[cache_key] = simulation

        model_results: dict[str, AggregateResult] = {}
        for model in experiment.models:
            pipeline_config = PipelineConfig(
                simulator=experiment.simulator,
                model=model,
                sampling=experiment.sampling,
                training=training_config,
                evaluation=evaluation_config,
            )
            model_results[model.label] = run_model_over_seeds(
                pipeline_config,
                experiment.seeds,
                simulation,
                parallel_config=parallel_config,
            )

        results[experiment.name] = model_results
        artifact["experiment_order"].append(experiment.name)
        artifact["experiments"][experiment.name] = build_artifact_profile(
            experiment=experiment,
            model_results=model_results,
            simulation=simulation,
            training_config=training_config,
            evaluation_config=evaluation_config,
        )
        save_experiment_artifact(artifact, save_results_path)
        print(f"[SAVED] Computation artifact updated at '{save_results_path}'.")

    return results


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
    header = (
            f"{'Experiment':<16} {'Model':<14} "
            + " ".join(f"{column:>17}" for column in columns)
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


def build_paper_experiments(
        seeds: tuple[int, ...] = (6, 42, 79, 123, 456),
) -> tuple[ExperimentSpec, ...]:
    baseline = ModelConfig(
        label="Baseline",
        kernel_type="baseline",
        spatial_kernel="rbf",
    )
    sd_rbf = ModelConfig(
        label="SD-RBF",
        kernel_type="SD",
        spatial_kernel="rbf",
    )
    common_simulator = {"L": 10.0, "N": 60, "T": 5.5, "dt": 0.05}
    common_sampling = {
        "method": "stratified",
        "left_region": (0.0, 2.0),
        "middle_region": (2.0, 8.0),
        "right_region": (8.0, 10.0),
        "weights": (0.30, 0.40, 0.30),
        "include_domain_boundaries": True,
    }

    return (
        ExperimentSpec(
            name="periodic",
            row_label="Periodic",
            simulator=SimulatorConfig(
                **common_simulator,
                stiffness_profile="periodic",
                num_hinges=4,
            ),
            sampling=SamplingConfig(**common_sampling, train_fraction=0.06),
            models=(
                baseline,
                sd_rbf,
                ModelConfig(
                    label="SD-matched",
                    kernel_type="SD",
                    spatial_kernel="periodic",
                ),
            ),
            seeds=seeds,
        ),
        ExperimentSpec(
            name="C2",
            row_label=r"$C^2$ bumps",
            simulator=SimulatorConfig(
                **common_simulator,
                stiffness_profile="C2",
                num_hinges=8,
            ),
            sampling=SamplingConfig(**common_sampling, train_fraction=0.06),
            models=(
                baseline,
                sd_rbf,
                ModelConfig(
                    label="SD-matched",
                    kernel_type="SD",
                    spatial_kernel="matern",
                    matern_nu=2.5,
                ),
            ),
            seeds=seeds,
        ),
        ExperimentSpec(
            name="piecewise",
            row_label="Semi-piecewise",
            simulator=SimulatorConfig(
                **common_simulator,
                stiffness_profile="piecewise",
                num_hinges=2,
            ),
            sampling=SamplingConfig(**common_sampling, train_fraction=0.04),
            models=(
                baseline,
                sd_rbf,
                ModelConfig(
                    label="SD-matched",
                    kernel_type="SD",
                    spatial_kernel="changepoint",
                    changepoint_location=5.0,
                    changepoint_width=0.15,
                ),
            ),
            seeds=seeds,
        ),
    )


def compute_main_experiment(
        save_results_path: str | Path = (
                "./data/paper_experiment.pkl"
        ),
        parallel_config: ParallelConfig | None = None,
) -> dict[str, dict[str, AggregateResult]]:
    """Run the paper experiment and save data required for later plots."""
    parallel_config = (
            parallel_config or ParallelConfig(enabled=True, max_workers=5,
                                              torch_threads_per_worker=None, start_method="spawn",)
    )

    results = run_experiment_matrix(
        experiment_specs=build_paper_experiments(),
        save_results_path=save_results_path,
        parallel_config=parallel_config,
    )

    print_aggregate_table(results)
    return results


if __name__ == "__main__":
    compute_main_experiment(save_results_path="./data/paper_experiment.pkl")
