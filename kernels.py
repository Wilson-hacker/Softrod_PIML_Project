from __future__ import annotations

import gpytorch
import torch

from config import (
    INPUT_VARIABLE_INDICES,
    Q_INDEX,
    Z_INDEX,
    InputVariable,
    ModelConfig,
    ReducedBaseKernel,
)


class BaselineRBFKernel(gpytorch.kernels.Kernel):
    """ARD-RBF covariance on standardized inputs (p, q, z)."""

    def __init__(self, **kwargs):
        """Initialize an ARD-RBF kernel on all three input variables."""
        super().__init__(**kwargs)
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Evaluate the covariance on the full input state."""
        return self.base_kernel(x1, x2, diag=diag, **params)


class ReducedInputRBFKernel(gpytorch.kernels.Kernel):
    """RBF or linear covariance on selected input variables."""

    def __init__(
        self,
        variables: tuple[InputVariable, ...],
        base_kernel_type: ReducedBaseKernel = "rbf",
        **kwargs,
    ):
        """Initialize an RBF or linear kernel on the selected variables."""
        super().__init__(**kwargs)
        active_dims = tuple(INPUT_VARIABLE_INDICES[variable] for variable in variables)
        if base_kernel_type == "rbf":
            self.base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=len(variables), active_dims=active_dims
            )
        else:
            self.base_kernel = gpytorch.kernels.LinearKernel(active_dims=active_dims)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Evaluate the covariance on the selected input variables."""
        return self.base_kernel(x1, x2, diag=diag, **params)


class StructuralDecoupledKernel(gpytorch.kernels.Kernel):
    """Product covariance k_q(q, q') * k_z(z, z')."""

    def __init__(
        self,
        model_config: ModelConfig,
        **kwargs,
    ):
        """Initialize the configured strain and spatial kernels."""
        super().__init__(**kwargs)

        if model_config.strain_kernel == "rbf":
            self.strain_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        elif model_config.strain_kernel == "linear":
            self.strain_kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unknown strain kernel: {model_config.strain_kernel!r}.")

        spatial_kernel = model_config.spatial_kernel

        if spatial_kernel == "rbf":
            self.spatial_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        elif spatial_kernel == "periodic":
            self.spatial_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=1)
        elif spatial_kernel == "matern":
            self.spatial_kernel = gpytorch.kernels.MaternKernel(
                nu=model_config.matern_nu,
                ard_num_dims=1,
            )
        elif spatial_kernel == "changepoint":
            self.spatial_kernel = ChangepointSpatialKernel(
                init_location=model_config.changepoint_location,
                init_width=model_config.changepoint_width,
                left_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=1),
                right_kernel=gpytorch.kernels.RBFKernel(ard_num_dims=1),
                learn_location=True,
                learn_width=True,
            )
        elif spatial_kernel == "warped":
            self.spatial_kernel = GibbsSpatialKernel(
                n_freqs=model_config.n_freqs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown spatial kernel: {spatial_kernel!r}.")

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Multiply the strain and spatial covariances."""
        q1 = x1[..., Q_INDEX : Q_INDEX + 1]
        z1 = x1[..., Z_INDEX : Z_INDEX + 1]
        q2 = x2[..., Q_INDEX : Q_INDEX + 1]
        z2 = x2[..., Z_INDEX : Z_INDEX + 1]

        covariance_strain = self.strain_kernel(q1, q2, diag=diag, **params)
        covariance_space = self.spatial_kernel(z1, z2, diag=diag, **params)

        return covariance_strain * covariance_space


class GibbsStateKernel(gpytorch.kernels.Kernel):
    def __init__(self, param_dict, **kwargs):
        """Initialize a spatial warp and an RBF kernel on the full state."""
        super().__init__(**kwargs)
        self.n_freqs = param_dict["n_freqs"]
        # Apply the RBF to (p, q, z_warped).
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
        self.spatial_kernel = GibbsSpatialKernel(n_freqs=self.n_freqs, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Inputs have shape (..., N, 3) with columns (p, q, z).
        """Evaluate the state covariance after warping spatial coordinates."""
        p1, q1, z1 = x1[..., 0:1], x1[..., 1:2], x1[..., 2:3]
        p2, q2, z2 = x2[..., 0:1], x2[..., 1:2], x2[..., 2:3]

        z1_warped, z2_warped = self.spatial_kernel.get_warped_z(z1, z2)

        # Combine the original state variables with warped coordinates.
        v1 = torch.cat([p1, q1, z1_warped], dim=-1)
        v2 = torch.cat([p2, q2, z2_warped], dim=-1)

        return self.base_kernel(v1, v2, diag=diag, **params)


class GibbsSpatialKernel(gpytorch.kernels.Kernel):
    def __init__(self, n_freqs, **kwargs):
        """Initialize the Fourier spatial scale and one-dimensional RBF kernel."""
        super().__init__(**kwargs)
        self.n_freqs = n_freqs
        # Apply the RBF to the warped spatial coordinate.
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)

        # Parameterize the spatial scale with a Fourier series.
        self.register_parameter("c", torch.nn.Parameter(torch.tensor([1.0])))
        self.register_parameter("a_coeffs", torch.nn.Parameter(torch.randn(n_freqs)))
        self.register_parameter("b_coeffs", torch.nn.Parameter(torch.randn(n_freqs)))
        self.register_parameter("w", torch.nn.Parameter(torch.tensor([1.0])))

    def get_warped_z(self, x1, x2):
        """Warp both coordinate sets using the learned spatial scale."""

        def get_ell(z):
            """Evaluate the positive spatial scale from its Fourier coefficients."""
            fourier_sum = 0
            for k in range(1, self.n_freqs + 1):
                fourier_sum += self.a_coeffs[k - 1] * torch.cos(k * self.w * z)
                fourier_sum += self.b_coeffs[k - 1] * torch.sin(k * self.w * z)
            return torch.abs(self.c + fourier_sum) + 1e-4

        z1_warped = x1 / get_ell(x1)
        z2_warped = x2 / get_ell(x2)
        return z1_warped, z2_warped

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Evaluate the RBF covariance on warped spatial coordinates."""
        z1_warped, z2_warped = self.get_warped_z(x1, x2)
        return self.base_kernel(z1_warped, z2_warped, diag=diag, **params)


class ChangepointSpatialKernel(gpytorch.kernels.Kernel):
    """Blend two spatial kernels with a sigmoid gate.

    For s(z) = sigmoid((z - z0) / width), the covariance is
    (1 - s(z)) * (1 - s(z')) * k_left(z, z')
    + s(z) * s(z') * k_right(z, z').
    """

    raw_location: torch.nn.Parameter
    raw_width: torch.nn.Parameter

    def __init__(
        self,
        left_kernel=None,
        right_kernel=None,
        init_location=0.0,
        init_width=0.5,
        learn_location=True,
        learn_width=True,
        **kwargs,
    ):
        """Initialize the component kernels and sigmoid gate parameters."""
        super().__init__(**kwargs)

        self.left_kernel = (
            left_kernel
            if left_kernel is not None
            else gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.right_kernel = (
            right_kernel
            if right_kernel is not None
            else gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )

        location = torch.tensor(float(init_location))
        unconstrained_width = torch.log(torch.expm1(torch.tensor(float(init_width))))

        self.raw_location = torch.nn.Parameter(location)
        self.raw_width = torch.nn.Parameter(unconstrained_width)

        self.raw_location.requires_grad_(learn_location)
        self.raw_width.requires_grad_(learn_width)

    @property
    def location(self):
        """Return the changepoint location."""
        return self.raw_location

    @property
    def width(self):
        """Return the positive transition width."""
        return torch.nn.functional.softplus(self.raw_width) + 1e-4

    def gate(self, z):
        """Evaluate the sigmoid weight for the right-hand kernel."""
        return torch.sigmoid((z - self.location) / self.width)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Blend the component covariances using the changepoint gate."""
        s1 = self.gate(x1)
        s2 = self.gate(x2)

        k_left = self.left_kernel(x1, x2, diag=diag, **params)
        k_right = self.right_kernel(x1, x2, diag=diag, **params)

        if diag:
            s1_flat = s1.squeeze(-1)
            s2_flat = s2.squeeze(-1)

            return (1.0 - s1_flat) * (
                1.0 - s2_flat
            ) * k_left + s1_flat * s2_flat * k_right

        return (1.0 - s1) * (1.0 - s2.transpose(-2, -1)) * k_left + s1 * s2.transpose(
            -2, -1
        ) * k_right


def build_kernel(model_config: ModelConfig) -> gpytorch.kernels.Kernel:
    """Build the configured kernel with a learnable outer output scale."""
    if model_config.kernel_type == "rbf":
        base_kernel: gpytorch.kernels.Kernel = BaselineRBFKernel()
    elif model_config.kernel_type == "reduced_input":
        base_kernel: gpytorch.kernels.Kernel = ReducedInputRBFKernel(
            variables=model_config.reduced_input_variables,
            base_kernel_type=model_config.reduced_input_base_kernel,
        )
    elif model_config.kernel_type == "SD":
        base_kernel = StructuralDecoupledKernel(model_config)
    elif model_config.kernel_type == "warped":
        base_kernel = GibbsStateKernel({"n_freqs": model_config.n_freqs})
    else:
        raise ValueError(f"Unknown kernel type: {model_config.kernel_type!r}.")

    return gpytorch.kernels.ScaleKernel(base_kernel)
