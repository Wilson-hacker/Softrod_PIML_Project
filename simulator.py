import numpy as np
from scipy.integrate import solve_ivp


class SoftRodSimulator:
    def __init__(self, L=10.0, N=100, T=5.0, dt=0.01, num_hinges=4):
        self.L = L
        self.N = N
        self.T = T
        self.dt = dt
        self.z = np.linspace(0, L, N)
        self.dz = self.z[1] - self.z[0]
        self.damping = 0.05

        # Geometry: varying stiffness k(z) for "thin hinges"
        hinge_freq = num_hinges * np.pi / L
        self.stiffness = 10.0 + 4.0 * np.cos(2 * hinge_freq * self.z)

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

    def generate_dataset(self):
        q0 = -4.0 * (self.z - self.L / 2) * np.exp(-(self.z - self.L / 2) ** 2)
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