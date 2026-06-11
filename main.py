import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator import SoftRodSimulator
from model import train_gp_model
from utils import plot_comparison, simulate_gp_phs, plot_time_slices


def run_pipeline():
    print("\n--- Step 1: Solving Soft Robot PDE via Numerical Integrators ---")
    sim = SoftRodSimulator(L=10.0, N=60, T=5.5, dt=0.05, num_hinges=4)
    data = sim.generate_dataset()

    p_flat = data['p'].flatten()
    q_flat = data['q'].flatten()
    dp_int_flat = data['dp_int'].flatten()
    dq_int_flat = data['dq_int'].flatten()

    # NEW: Tile the spatial array to match the time dimension, then flatten
    N_pts, T_steps = data['q'].shape
    z_tiled = np.tile(data['z'].reshape(N_pts, 1), (1, T_steps))
    z_flat = z_tiled.flatten()

    # NEW: X matrix is now shape [N_total, 3] -> (p, q, z)
    X = np.stack([p_flat, q_flat, z_flat], axis=1)

    downscale_factor = 10
    X_train = torch.tensor(X[::downscale_factor], dtype=torch.float32)
    y_dEdp_train = torch.tensor(dq_int_flat[::downscale_factor], dtype=torch.float32)
    y_dEdq_train = torch.tensor(dp_int_flat[::downscale_factor], dtype=torch.float32)

    print(f"Data Generation Complete. GP Input Shape: {X_train.shape}")

    # NEW: Z-Score Normalization
    X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-6)

    y_p_mean, y_p_std = y_dEdp_train.mean(), y_dEdp_train.std()
    y_p_train_norm = (y_dEdp_train - y_p_mean) / (y_p_std + 1e-6)

    y_q_mean, y_q_std = y_dEdq_train.mean(), y_dEdq_train.std()
    y_q_train_norm = (y_dEdq_train - y_q_mean) / (y_q_std + 1e-6)

    print("\n--- Step 2: Training GP-PHS Component Models ---")
    dEdp_model, dEdp_lh = train_gp_model(X_train_norm, y_p_train_norm, kernel_type="baseline", training_iter=1000, patience=15)
    dEdq_model, dEdq_lh = train_gp_model(X_train_norm, y_q_train_norm, kernel_type="baseline", training_iter=1000, patience=15)

    # Store normalization stats directly on the models for easy access in utils.py
    dEdp_model.X_mean, dEdp_model.X_std = X_mean, X_std
    dEdp_model.y_mean, dEdp_model.y_std = y_p_mean, y_p_std
    dEdq_model.X_mean, dEdq_model.X_std = X_mean, X_std
    dEdq_model.y_mean, dEdq_model.y_std = y_q_mean, y_q_std

    print("\n--- Step 3: Projecting and Evaluating System Approximations ---")
    plot_comparison(data, dEdp_model, dEdq_model, sim, save_path="./plots/gpphs_comparison.png")

    print("\n--- Step 4: Running Dynamic Rollout and Time-Slice Comparisons ---")
    p_gp, q_gp = simulate_gp_phs(sim, dEdp_model, dEdq_model)
    plot_time_slices(data, q_gp, sim, time_slices=[0, 50, 100], save_path="./plots/gpphs_string_slices.png")
    print("Pipeline execution completed successfully.\n")


if __name__ == '__main__':
    run_pipeline()