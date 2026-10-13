# Structure-Aware Spatial Kernels for GP-PHS

Code for **Structure-Aware Spatial Kernels for Gaussian Process Distributed
Port-Hamiltonian Systems** by Tianyi Yang, Thomas Beckers, Mohammad Ali, and
Kaiyuan Tan. Submission to the American Control Conference (ACC) is planned.

The experiments compare Gaussian process kernels for learning Hamiltonian
gradients in a one-dimensional soft rod with spatially varying stiffness.
Evaluation includes effort prediction, dynamical rollouts, energy error, and
integrability.

## Setup

Use Python 3.13. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
MP4 output also requires FFmpeg on your system path. The rollout code runs on CPU.

## Run the experiments

```bash
python run_experiments.py
```

This runs the experiment matrix defined in `build_paper_experiments()` and saves
results to `data/paper_experiment.pkl`. The defaults use five seeds and up to five
worker processes. Edit `build_paper_experiments()` to change profiles, kernels,
sampling, or training settings; edit `ParallelConfig` in
`compute_main_experiment()` to adjust parallel execution. Configuration fields
are defined in `config.py`.

To generate the metric table, rollout figure, and example animation from saved
results:

```bash
python report.py
```

The default outputs are `plots/rollout_experiments.pdf` and
`plots/C2_baseline3.mp4`. Edit the final block of `report.py` to select the input
artifact, figures, and animation. If you only need the table and PDF, omit the
`create_rollout_video_from_artifact()` call; FFmpeg is then unnecessary.

Run both scripts from the repository root. Reporting reuses saved results and
does not retrain the models.

## Repository structure

| File or directory | Purpose |
| --- | --- |
| `run_experiments.py` | Experiment definitions, training, evaluation, and result saving |
| `report.py` | Tables, paper figures, and example animation |
| `config.py` | Simulation, model, training, and evaluation settings |
| `simulator.py` | Reference soft-rod dynamics and stiffness profiles |
| `kernels.py`, `model.py` | GP kernels and model training |
| `data_processing.py` | Training-point sampling and normalization |
| `rollout.py` | GP-PHS integration and displacement reconstruction |
| `metrics.py`, `plotting.py` | Evaluation metrics and visualization |
| `utils.py` | Earlier standalone analysis utilities |
| `data/`, `plots/` | Generated results and figures (excluded from Git) |

Contact: [Tianyi Yang](mailto:tianyi.yang@vanderbilt.edu), Vanderbilt University.
