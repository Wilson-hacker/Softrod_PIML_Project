from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

from config import SimulatorConfig

StiffnessFunction = Callable[[SimulatorConfig, np.ndarray], np.ndarray]


def const_stiffness(z_arr):
    return np.ones_like(z_arr) * 10.0


def periodic_stiffness(num_hinges, length, z_arr):
    hinge_freq = num_hinges * np.pi / length
    return 10.0 + 4.0 * np.cos(2 * hinge_freq * z_arr)


def periodic_c2_bumps(
    num_bumps: int,
    length: float,
    z_arr: np.ndarray,
    base: float = 10.0,
    amplitude: float = 4.0,
    radius: float | None = None,
) -> np.ndarray:
    """Construct deterministic compactly supported Wendland C2 bumps."""
    z_arr = np.asarray(z_arr, dtype=float)

    if radius is None:
        radius = 0.5 * length / num_bumps
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    centers = np.linspace(0.1 * length, 0.9 * length, num_bumps)
    weights = (-1.0) ** np.arange(num_bumps)

    normalized_distance = np.abs(z_arr[:, None] - centers[None, :]) / radius
    basis = np.where(
        normalized_distance < 1.0,
        (1.0 - normalized_distance) ** 4 * (4.0 * normalized_distance + 1.0),
        0.0,
    )
    profile = basis @ weights

    max_abs = float(np.max(np.abs(profile)))
    if max_abs > 1e-12:
        profile = profile / max_abs

    return base + amplitude * profile


def c2_multiscale_stiffness(
    num_bumps: int,
    length: float,
    z_arr: np.ndarray,
    k_min: float = 5.0,
    k_max: float = 18.0,
    seed: int = 42,
    min_radius_frac: float = 0.035,
    max_radius_frac: float = 0.12,
) -> np.ndarray:
    """Construct a fixed randomized multiscale Wendland-C2 profile."""
    if num_bumps <= 0:
        raise ValueError("num_bumps must be positive.")
    if not 0.0 < min_radius_frac <= max_radius_frac:
        raise ValueError("Radius fractions must satisfy 0 < min <= max.")
    if k_min >= k_max:
        raise ValueError("k_min must be smaller than k_max.")

    z_arr = np.asarray(z_arr, dtype=float)
    rng = np.random.default_rng(seed)

    centers = rng.uniform(0.0, length, size=num_bumps)
    radii = rng.uniform(
        min_radius_frac * length,
        max_radius_frac * length,
        size=num_bumps,
    )
    weights = rng.normal(0.0, 1.0, size=num_bumps)

    normalized_distance = np.abs(z_arr[:, None] - centers[None, :]) / radii[None, :]
    basis = np.where(
        normalized_distance < 1.0,
        (1.0 - normalized_distance) ** 4 * (4.0 * normalized_distance + 1.0),
        0.0,
    )
    profile = basis @ weights

    profile += 0.5 * np.sin(2.0 * np.pi * z_arr / length)
    profile += 0.3 * np.cos(4.0 * np.pi * z_arr / length)

    profile_min = float(np.min(profile))
    profile_max = float(np.max(profile))

    if profile_max - profile_min < 1e-12:
        return np.full_like(z_arr, 0.5 * (k_min + k_max))

    profile_01 = (profile - profile_min) / (profile_max - profile_min)
    return k_min + (k_max - k_min) * profile_01


def piecewise_stiffness(
    length: float,
    z_arr: np.ndarray,
    values: tuple[float, float] = (5.0, 15.0),
    transition_center: float | None = None,
    transition_width: float | None = None,
    smoothness: str = "C2",
) -> np.ndarray:
    """Construct a two-region stiffness profile with a smooth transition."""
    z_arr = np.asarray(z_arr, dtype=float)
    k_left, k_right = values

    if transition_center is None:
        transition_center = 0.5 * length

    if transition_width is None:
        spatial_steps = np.diff(np.sort(z_arr))
        if spatial_steps.size == 0:
            raise ValueError("z_arr must contain at least two points.")
        transition_width = 2.0 * float(np.median(spatial_steps))

    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive.")

    z_start = transition_center - 0.5 * transition_width
    normalized_coordinate = np.clip(
        (z_arr - z_start) / transition_width,
        0.0,
        1.0,
    )

    smoothness_upper = smoothness.upper()
    if smoothness_upper == "C1":
        weight = 3.0 * normalized_coordinate**2 - 2.0 * normalized_coordinate**3
    elif smoothness_upper == "C2":
        weight = (
            6.0 * normalized_coordinate**5
            - 15.0 * normalized_coordinate**4
            + 10.0 * normalized_coordinate**3
        )
    else:
        raise ValueError("smoothness must be either 'C1' or 'C2'.")

    return k_left + (k_right - k_left) * weight


def sharp_transition_stiffness(
    length, z_arr, low=6.0, high=14.0, center=None, width=0.1
):
    """
    Smooth but sharp transition between two stiffness levels.

    Regularity: smooth for finite width, but approaches a discontinuous
    step as width -> 0.
    Useful for testing changepoint kernels and robustness to sharp interfaces.
    """
    z_arr = np.asarray(z_arr)

    if center is None:
        center = 0.5 * length

    transition = 0.5 * (1.0 + np.tanh((z_arr - center) / width))

    return low * (1.0 - transition) + high * transition


def build_stiffness_profile(
    config: SimulatorConfig,
    z_arr: np.ndarray,
) -> np.ndarray:
    """Construct the configured material stiffness field."""
    profile = config.stiffness_profile

    if profile == "constant":
        stiffness = const_stiffness(z_arr)
    elif profile == "periodic":
        stiffness = periodic_stiffness(config.num_hinges, config.L, z_arr)
    elif profile == "periodic bumps":
        stiffness = periodic_c2_bumps(
            config.num_hinges,
            config.L,
            z_arr,
        )
    elif profile == "C2":
        stiffness = c2_multiscale_stiffness(
            config.num_hinges,
            config.L,
            z_arr,
            seed=config.profile_seed,
        )
    elif profile == "piecewise":
        if config.num_hinges != 2:
            raise ValueError("The current piecewise profile requires num_hinges=2.")
        stiffness = piecewise_stiffness(config.L, z_arr)
    elif profile == "sharp":
        stiffness = sharp_transition_stiffness(config.L, z_arr)
    else:
        raise ValueError(f"Unknown stiffness profile: {profile!r}.")

    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != z_arr.shape:
        raise ValueError("The stiffness profile must match the spatial grid.")
    if not np.isfinite(stiffness).all():
        raise ValueError("The stiffness profile contains nonfinite values.")
    if np.any(stiffness <= 0.0):
        raise ValueError("The stiffness profile must remain strictly positive.")

    return stiffness


class SoftRodSimulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.L = config.L
        self.N = config.N
        self.T = config.T
        self.dt = config.dt
        self.damping = config.damping
        self.profile_seed = config.profile_seed
        self.stiffness_profile = config.stiffness_profile
        self.num_hinges = config.num_hinges

        self.z = np.linspace(0.0, self.L, self.N)
        self.dz = float(self.z[1] - self.z[0])
        self.stiffness = build_stiffness_profile(config, self.z)

        # This spatial factor is constant throughout all ODE evaluations.
        self.motor_location = np.exp(-10.0 * (self.z - self.L / 2.0) ** 2)

    def dEdq(self, q: np.ndarray) -> np.ndarray:
        """Evaluate dH/dq for q with shape (N,) or (N,T)."""
        q = np.asarray(q, dtype=float)

        if q.ndim == 1:
            if q.shape[0] != self.N:
                raise ValueError("A one-dimensional q must have length N.")
            stiffness = self.stiffness
        elif q.ndim == 2:
            if q.shape[0] != self.N:
                raise ValueError("The first dimension of q must equal N.")
            stiffness = self.stiffness[:, None]
        else:
            raise ValueError("q must have shape (N,) or (N,T).")

        return stiffness * q + 2.0 * q**3

    def dEdp(self, p):
        """Derivative of kinetic energy w.r.t momentum"""
        return np.asarray(p, dtype=float)

    def motor_force(self, t, z):
        """
        Time-dependent external force u(t).
        Simulates motors embedded at specific locations along the rod,
        pulsing energy into the system over time.
        """
        # Example: A motor at the middle of the rod that pulses periodically
        motor_pulse = np.sin(2.0 * np.pi * t)
        return self.motor_location * motor_pulse

    def evaluate_ph_dynamics(self, p, q, grad_H_p, grad_H_q, t):
        """
        The core Port-Hamiltonian structural physics mapping:
        dx/dt = (J - R) * dH/dx + G * u(t)
        """
        dp_dt = np.zeros_like(p)
        dq_dt = np.zeros_like(q)

        # J matrix (Interconnection via spatial derivative)
        dq_dt[1:-1] = (grad_H_p[2:] - grad_H_p[:-2]) / (2 * self.dz)
        dp_dt[1:-1] = (grad_H_q[2:] - grad_H_q[:-2]) / (2 * self.dz)

        # R matrix (Dissipation)
        dp_dt -= self.damping * grad_H_p

        # EXTERNAL INPUT: Adding the time-dependent motor force to momentum
        dp_dt += self.motor_force(t, self.z)

        # Boundary conditions (Fixed ends)
        dq_dt[[0, -1]] = 0.0
        dp_dt[[0, -1]] = 0.0

        return np.concatenate([dp_dt, dq_dt])

    def pde_rhs(self, t, state):
        """Standard RHS using true analytical energy gradients."""
        p = state[: self.N]
        q = state[self.N :]

        return self.evaluate_ph_dynamics(
            p, q, grad_H_p=self.dEdp(p), grad_H_q=self.dEdq(q), t=t
        )

    def generate_dataset(self, get_init_q_config):
        q0 = get_init_q_config(self.z, self.L)
        p0 = np.zeros(self.N)
        state0 = np.concatenate([p0, q0])

        t_eval = np.arange(0, self.T, self.dt)
        sol = solve_ivp(self.pde_rhs, [0, self.T], state0, t_eval=t_eval, method="RK45")

        if not sol.success:
            raise RuntimeError(f"Reference simulation failed: {sol.message}")
        if sol.y.shape[1] != t_eval.size:
            raise RuntimeError(
                "Reference simulation returned fewer output times than requested."
            )

        p_data = sol.y[: self.N, :]
        q_data = sol.y[self.N :, :]

        grad_H_p = self.dEdp(p_data)
        grad_H_q = self.dEdq(q_data)

        return {
            "p": p_data,
            "q": q_data,
            "dH_dp": grad_H_p,
            "dH_dq": grad_H_q,
            "t": sol.t,
            "z": self.z,
        }

    def energy_density(self, p, q):
        """
        Hamiltonian density:
            h(p,q,z) = 1/2 p^2 + 1/2 k(z) q^2 + 1/2 q^4

        Supports:
            p, q shape [N] or [N, T]
        """
        k = self.stiffness

        if p.ndim == 1:
            return 0.5 * p**2 + 0.5 * k * q**2 + 0.5 * q**4

        if p.ndim == 2:
            return 0.5 * p**2 + 0.5 * k[:, None] * q**2 + 0.5 * q**4

        raise ValueError(f"Expected p, q to have ndim 1 or 2, got {p.ndim}.")

    def total_energy(self, p, q):
        """
        Spatial integral of the Hamiltonian density.

        Returns:
            scalar if p, q are [N]
            array of shape [T] if p, q are [N, T]
        """
        density = self.energy_density(p, q)
        return np.trapezoid(density, self.z, axis=0)
