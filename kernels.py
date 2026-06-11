import torch
import gpytorch


class BaselineRBFKernel(gpytorch.kernels.Kernel):
    """Standard stationary RBF kernel (your baseline to beat)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x1, x2, diag=False, **params):
        return self.rbf(x1, x2, diag=diag, **params)


class HingeAwareStateKernel(gpytorch.kernels.Kernel):
    """
    YOUR RESEARCH GOES HERE.
    A composite kernel that adapts its lengthscale based on the state 'q'
    (e.g., when strain 'q' is high, we are likely at a hinge, so change covariance).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Example structure: combining a linear kernel for rigid parts
        # and an RBF for the highly nonlinear hinge parts.
        self.linear = gpytorch.kernels.LinearKernel()
        self.rbf = gpytorch.kernels.RBFKernel()

        # You will define a state-dependent warping or scaling function here
        self.register_parameter(
            name="hinge_sensitivity",
            parameter=torch.nn.Parameter(torch.tensor([1.0]))
        )

    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 shape: [batch, 2] where col 0 is p, col 1 is q
        q1 = x1[:, 1].unsqueeze(-1)
        q2 = x2[:, 1].unsqueeze(-1)

        # Placeholder for your math: e.g., modulating RBF variance based on strain magnitude
        scaling_1 = torch.tanh(self.hinge_sensitivity * q1 ** 2)
        scaling_2 = torch.tanh(self.hinge_sensitivity * q2 ** 2)

        # This is just a conceptual placeholder. You will write the actual
        # state-dependent covariance math here.
        k_base = self.rbf(x1, x2, diag=diag, **params)

        if diag:
            return k_base * (scaling_1.squeeze() * scaling_2.squeeze())
        else:
            return k_base * (scaling_1 @ scaling_2.T)