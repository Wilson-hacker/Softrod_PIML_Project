import numpy as np
from scipy.integrate import solve_ivp


def periodic_stiffness(num_hinges, length, z_arr):
    hinge_freq = num_hinges * np.pi / length
    return 10.0 + 4.0 * np.cos(2 * hinge_freq * z_arr)


# def c2_less_smooth_stiffness(num_bumps, length, z_arr, base=10.0, amp=4.0, radius=None):
#     """
#     Less-smooth but still C^2 stiffness profile using compactly supported
#     Wendland C^2 basis functions.
#
#     Regularity: C^2, but generally not C^3.
#     Useful for testing Matérn kernels with finite smoothness assumptions.
#
#     k(z) = base + amp * normalized_sum_of_C2_bumps(z)
#     """
#     z_arr = np.asarray(z_arr)
#
#     if radius is None:
#         radius = 0.5 * length / num_bumps
#
#     centers = np.linspace(0.1 * length, 0.9 * length, num_bumps)
#
#     # Alternating bump amplitudes to create spatial heterogeneity.
#     weights = np.array([(-1.0) ** i for i in range(num_bumps)])
#
#     profile = np.zeros_like(z_arr, dtype=float)
#
#     for center, weight in zip(centers, weights):
#         r = np.abs(z_arr - center) / radius
#
#         # Wendland C^2 function in 1D:
#         # phi(r) = (1-r)^4_+ (4r+1)
#         phi = np.where(
#             r < 1.0,
#             (1.0 - r) ** 4 * (4.0 * r + 1.0),
#             0.0,
#         )
#
#         profile += weight * phi
#
#     # Normalize to roughly [-1, 1] before scaling.
#     max_abs = np.max(np.abs(profile))
#     if max_abs > 1e-12:
#         profile = profile / max_abs
#
#     return base + amp * profile


def c2_multiscale_stiffness(
    num_bumps,
    length,
    z_arr,
    k_min=5.0,
    k_max=18.0,
    seed=42,
    min_radius_frac=0.035,
    max_radius_frac=0.12,
):
    """
    Rough but C^2 stiffness profile using compactly supported Wendland C^2 bumps.

    Regularity:
        C^2, but not generally C^3.

    Why this is harder than the previous version:
        - more bumps,
        - random centers,
        - multiple spatial scales,
        - variation distributed across the full domain,
        - rescaled to a fixed positive stiffness range [k_min, k_max].
    """
    z_arr = np.asarray(z_arr)
    rng = np.random.default_rng(seed)

    centers = rng.uniform(0.0, length, size=num_bumps)
    radii = rng.uniform(
        min_radius_frac * length,
        max_radius_frac * length,
        size=num_bumps,
    )
    weights = rng.normal(loc=0.0, scale=1.0, size=num_bumps)

    profile = np.zeros_like(z_arr, dtype=float)

    for center, radius, weight in zip(centers, radii, weights):
        r = np.abs(z_arr - center) / radius

        # Wendland C^2:
        # phi(r) = (1-r)^4_+ (4r+1)
        phi = np.where(
            r < 1.0,
            (1.0 - r) ** 4 * (4.0 * r + 1.0),
            0.0,
        )

        profile += weight * phi

    # Add a mild low-frequency component so the whole domain is affected.
    profile += 0.5 * np.sin(2.0 * np.pi * z_arr / length)
    profile += 0.3 * np.cos(4.0 * np.pi * z_arr / length)

    # Rescale to [k_min, k_max].
    p_min = np.min(profile)
    p_max = np.max(profile)

    if p_max - p_min < 1e-12:
        return np.full_like(z_arr, 0.5 * (k_min + k_max))

    profile_01 = (profile - p_min) / (p_max - p_min)
    stiffness = k_min + (k_max - k_min) * profile_01

    return stiffness


def piecewise_stiffness(num_segments, length, z_arr, values=None):
    """
    Piecewise-constant stiffness profile.

    Regularity: discontinuous at segment boundaries.
    Useful for testing changepoint or piecewise kernels.
    """
    z_arr = np.asarray(z_arr)

    if values is None:
        # Deterministic alternating stiffness values.
        base_values = np.array([6.0, 14.0, 8.0, 12.0])
        values = np.resize(base_values, num_segments)
    else:
        values = np.asarray(values, dtype=float)
        if len(values) != num_segments:
            raise ValueError("Length of values must match num_segments.")

    edges = np.linspace(0.0, length, num_segments + 1)

    # idx tells which segment each z belongs to.
    idx = np.digitize(z_arr, edges[1:-1], right=False)

    return values[idx]


def sharp_transition_stiffness(length, z_arr, low=6.0, high=14.0, center=None, width=0.1):
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


class SoftRodSimulator:
    def __init__(self, L=10.0, N=100, T=5.0, dt=0.01, num_hinges=4, stiffness_profile="periodic"):
        self.L = L
        self.N = N
        self.T = T
        self.dt = dt
        self.z = np.linspace(0, L, N)
        self.dz = self.z[1] - self.z[0]
        self.damping = 0.05

        # Geometry: varying stiffness k(z) for "thin hinges"
        if stiffness_profile == "periodic":
            self.stiffness = periodic_stiffness(num_hinges, L, self.z)
        elif stiffness_profile == "C2":
            self.stiffness = c2_multiscale_stiffness(num_hinges, L, self.z)
        elif stiffness_profile == "piecewise":
            self.stiffness = piecewise_stiffness(num_hinges, L, self.z)
        elif stiffness_profile == "sharp":
            self.stiffness = sharp_transition_stiffness(L, self.z)
        else:
            raise ValueError(f"Unknown stiffness profile: {stiffness_profile}")


    def dEdq(self, q, z_idx):
        """Derivative of strain energy w.r.t strain"""
        k = self.stiffness[z_idx]
        return k * q + 2.0 * q ** 3

    def dEdp(self, p):
        """Derivative of kinetic energy w.r.t momentum"""
        return p

    def motor_force(self, t, z):
        """
        Time-dependent external force u(t).
        Simulates motors embedded at specific locations along the rod,
        pulsing energy into the system over time.
        """
        # Example: A motor at the middle of the rod that pulses periodically
        motor_location = np.exp(-10.0 * (z - self.L / 2) ** 2)
        motor_pulse = np.sin(2 * np.pi * t)
        return motor_location * motor_pulse

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
        dq_dt[0], dq_dt[-1] = 0.0, 0.0
        dp_dt[0], dp_dt[-1] = 0.0, 0.0

        return np.concatenate([dp_dt, dq_dt])

    def pde_rhs(self, t, state):
        """Standard RHS using true analytical energy gradients."""
        p = state[:self.N]
        q = state[self.N:]

        grad_H_p = self.dEdp(p)
        grad_H_q = np.array([self.dEdq(q[i], i) for i in range(self.N)])

        # Delegate to the structural physics operator
        return self.evaluate_ph_dynamics(p, q, grad_H_p, grad_H_q, t)

    def generate_dataset(self, get_init_q_config):
        q0 = get_init_q_config(self.z, self.L)
        # q0 = -4.0 * (self.z - self.L / 2) * np.exp(-(self.z - self.L / 2) ** 2)
        p0 = np.zeros(self.N)
        state0 = np.concatenate([p0, q0])

        t_eval = np.arange(0, self.T, self.dt)
        sol = solve_ivp(self.pde_rhs, [0, self.T], state0, t_eval=t_eval, method='RK45')

        p_data = sol.y[:self.N, :]
        q_data = sol.y[self.N:, :]

        grad_H_p = np.zeros_like(p_data)
        grad_H_q = np.zeros_like(q_data)

        # Bypass the unstable cumulative_trapezoid extraction.
        # Instead, we directly query the exact analytical energy gradients
        # to serve as pristine ground-truth targets for the GP.
        for i in range(len(sol.t)):
            p = sol.y[:self.N, i]
            q = sol.y[self.N:, i]

            grad_H_p[:, i] = self.dEdp(p)
            grad_H_q[:, i] = np.array([self.dEdq(q[j], j) for j in range(self.N)])

        return {
            'p': p_data, 'q': q_data,
            'dH_dp': grad_H_p, 'dH_dq': grad_H_q,
            't': sol.t, 'z': self.z
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
            return 0.5 * p ** 2 + 0.5 * k * q ** 2 + 0.5 * q ** 4

        if p.ndim == 2:
            return 0.5 * p ** 2 + 0.5 * k[:, None] * q ** 2 + 0.5 * q ** 4

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