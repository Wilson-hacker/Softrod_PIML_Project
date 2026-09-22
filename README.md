# Structure-Aware Spatial Kernels for GP-PHS

Code accompanying **Structure-Aware Spatial Kernels for Gaussian Process Port-Hamiltonian Systems**.

This repository implements and evaluates structure-aware Gaussian process kernels for learning energy-gradient models in port-Hamiltonian systems. The current experiments use a one-dimensional soft-rod/string system with spatially varying stiffness and compare multiple kernel designs through learned effort fields, dynamical rollouts, and physics-informed evaluation metrics.

## Overview

The project studies Gaussian process models for the Hamiltonian gradients

$$
\frac{\partial H}{\partial p}
\qquad \text{and} \qquad
\frac{\partial H}{\partial q},
$$

where $p$ and $q$ denote the momentum- and strain-like state variables of the system.

The main goal is to evaluate whether kernel structure that reflects the spatial and state dependence of the underlying physics improves model performance. The repository includes:

- baseline Gaussian process kernels;
- structure-aware and structurally decoupled kernel designs;
- configurable stiffness profiles;
- simulated training and evaluation data;
- GP-PHS rollouts using the learned Hamiltonian gradients;
- quantitative metrics for effort prediction, rollout accuracy, force-field error, and integrability;
- artifact-based plotting and reporting.

## Running the project

The workflow is split into two stages.

### 1. Run the experiments

```bash
python run_experiments.py
```

This command runs the configured experiment matrix and saves the numerical results to a serialized `.pkl` artifact.

The artifact contains the information needed for later analysis and visualization, including:

- experiment and model configuration;
- reference trajectories;
- predicted trajectories;
- aggregate metrics across random seeds;
- reconstructed displacement fields;
- force-field errors;
- metadata required by the reporting code.

The exact output path is defined in the experiment configuration or in `run_experiments.py`.

### 2. Generate reports and figures

```bash
python report.py
```

This command loads the saved `.pkl` artifact and generates the configured tables and visualizations.

Separating experiment execution from reporting allows the computationally expensive models to be trained once while figures and summaries can be regenerated independently.

## Installation

A recent Python 3 installation is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib torch gpytorch
```

Additional development or formatting dependencies may be listed in `requirements.txt`, `pyproject.toml`, or the repository documentation when available.

## Repository structure

A typical organization of the codebase is:

```text
.
├── run_experiments.py      # Runs the experiment matrix and writes a .pkl artifact
├── report.py               # Loads the artifact and generates figures/tables
├── config.py               # Dataclasses and experiment configuration
├── simulator.py            # Soft-rod/string simulator and stiffness profiles
├── kernels.py              # Baseline and structure-aware GP kernels
├── model.py                # GP model definition and training loop
├── data_processing.py      # Sampling and normalization
├── rollout.py              # GP-PHS rollout and displacement reconstruction
├── metrics.py              # Numerical and physics-informed evaluation metrics
├── plotting.py             # Artifact-based plotting functions
├── data/                   # Generated .pkl artifacts
└── plots/                  # Generated figures
```

The exact structure may differ slightly as the project evolves.

## Experiment configuration

Experiments are defined through configuration dataclasses. A typical experiment specifies:

- the stiffness profile;
- the spatial and temporal discretization;
- the kernel architecture;
- training hyperparameters;
- the training-point sampling strategy;
- the random seeds;
- the ODE solver configuration;
- the metrics and plots to generate.

The default experiment matrix can be edited in `run_experiments.py` or in the relevant configuration module.

## Kernel designs

The repository supports several kernel architectures, including:

### Baseline kernel

A conventional radial basis function kernel over the model inputs.

### Momentum-only kernel

A kernel that depends only on the momentum-like state variable where appropriate.

### Structurally decoupled kernel

A product kernel that separates state and spatial dependence, for example,

$$
k \left( (q,z),(q',z') \right) = k_q(q,q') \, k_z(z,z').
$$

Different spatial kernels can be used inside this construction, such as radial basis function, periodic, Matérn, or changepoint kernels.

### Warped or nonstationary kernel

A state- or position-dependent kernel design intended to represent nonstationary behavior.

The exact set of available kernels and their parameters is defined in `kernels.py` and `config.py`.

## Evaluation

The project evaluates the learned models using several complementary quantities.

### Effort prediction

Prediction errors for the learned Hamiltonian gradients:

$$
\frac{\partial H}{\partial p},
\qquad
\frac{\partial H}{\partial q}.
$$

### Dynamical rollout

The learned gradients are inserted into the port-Hamiltonian dynamics and integrated numerically. Predicted \(p\) and \(q\) trajectories are compared with the reference simulation.

### Displacement reconstruction

The displacement field is reconstructed from the predicted strain field using cumulative spatial integration.

### Force-field error

The spatial derivative of the predicted constitutive effort is compared with the corresponding reference quantity:

$$
\left| \partial_z\!\left(\frac{\partial H_{\mathrm{pred}}}{\partial q}\right) - \partial_z\!\left(\frac{\partial H_{\mathrm{true}}}{\partial q}\right)
\right|.
$$

### Curl/integrability error

The learned effort field is tested for consistency with a common scalar Hamiltonian through the mixed-partial condition

$$
\frac{\partial}{\partial q}
\left(\frac{\partial H}{\partial p}\right) = \frac{\partial}{\partial p}
\left(\frac{\partial H}{\partial q}\right).
$$

Reported summaries may include mean absolute curl, curl RMSE, and maximum absolute curl.

## Reproducibility

The experiment artifact stores the relevant numerical and model configuration so that results can be traced back to the settings used to generate them.

For reproducible comparisons:

- keep the training seeds fixed;
- keep the stiffness-profile seed fixed when applicable;
- use the same simulator and ODE-solver settings;
- record the Python and package versions;
- avoid mixing Conda and virtual-environment toolchains in the same shell;
- generate plots from the saved artifact rather than rerunning the experiment.

For publication, it is recommended to include an environment file such as one of the following:

```text
requirements.txt
pyproject.toml
environment.yml
```

## Outputs

The experiment runner produces a `.pkl` artifact containing the results for all configured experiments and models.

The reporting script can generate outputs such as:

- aggregate metric tables;
- rollout time-slice comparisons;
- multi-profile rollout grids;
- stiffness-profile visualizations;
- force-field error heatmaps;
- additional paper figures.

Generated files should normally be written to a dedicated directory such as `results/` or `figures/`.

## Extending the project

New experiments can be added by:

1. defining a new stiffness profile in `simulator.py`;
2. adding a kernel or spatial-kernel option in `kernels.py`;
3. exposing its parameters through the configuration dataclasses;
4. adding the corresponding `ExperimentSpec`;
5. running `python run_experiments.py`;
6. regenerating the report with `python report.py`.

## Paper

- Title: **Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems**
- Authors: Tianyi Yang, Thomas Beckers, Mohammad Ali, Kaiyuan Tan

To be submitted on ACC.

## Contact

For questions about the code or paper, contact:

**Tianyi Yang**  
*Department of Computer Science*, Vanderbilt University  
[Contact](mailto:tianyi.yang@vanderbilt.edu)
