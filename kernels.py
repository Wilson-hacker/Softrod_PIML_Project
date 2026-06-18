import torch
import gpytorch


class BaselineRBFKernel(gpytorch.kernels.Kernel):
    """Standard stationary RBF kernel (your baseline to beat)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rbf = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x1, x2, diag=False, **params):
        return self.rbf(x1, x2, diag=diag, **params)


class StructuralDecoupledKernel(gpytorch.kernels.Kernel):
    def __init__(self, spatial_kernel_type="rbf", n_freqs=3, **kwargs):
        super().__init__(**kwargs)
        self.strain_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        if spatial_kernel_type == "rbf":
            self.spatial_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
        elif spatial_kernel_type == "gibbs":
            self.spatial_kernel = GibbsSpatialKernel(nfreqs=n_freqs, **kwargs)
        else:
            raise ValueError(f"Unknown spatial kernel type: {spatial_kernel_type}")

    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 are [N, 2] -> col 0 is q, col 1 is z
        q1, z1 = x1[..., 1:2], x1[..., 2:3]
        q2, z2 = x2[..., 1:2], x2[..., 2:3]

        # Physics: Covariance is the product of strain similarity AND spatial similarity
        cov_strain = self.strain_kernel(q1, q2, diag=diag, **params)
        cov_spatial = self.spatial_kernel(z1, z2, diag=diag, **params)

        return cov_strain * cov_spatial


class GibbsStateKernel(gpytorch.kernels.Kernel):
    def __init__(self, n_freqs=3, **kwargs):
        super().__init__(**kwargs)
        self.n_freqs = n_freqs
        # The base kernel operates on the full 3D vector: (p, q, z_warped)
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
        self.spatial_kernel = GibbsSpatialKernel(nfreqs=n_freqs, **kwargs)


    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 shape: [..., N, 3] -> columns are (p, q, z)
        p1, q1, z1 = x1[..., 0:1], x1[..., 1:2], x1[..., 2:3]
        p2, q2, z2 = x2[..., 0:1], x2[..., 1:2], x2[..., 2:3]

        z1_warped, z2_warped = self.spatial_kernel.get_warped_z(p1, p2)

        # 3. Attach the transformed z back to the original (p, q) vectors
        v1 = torch.cat([p1, q1, z1_warped], dim=-1)
        v2 = torch.cat([p2, q2, z2_warped], dim=-1)

        # 4. Compute the final covariance using the full, reconstructed state
        return self.base_kernel(v1, v2, diag=diag, **params)


class GibbsSpatialKernel(gpytorch.kernels.Kernel):
    def __init__(self, n_freqs, **kwargs):
        super().__init__(**kwargs)
        self.n_freqs = n_freqs
        # Crucial: The base RBF here only looks at 1 dimension (the warped z)
        self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)

        # Fourier parameters for the spatial lengthscale
        self.register_parameter("c", torch.nn.Parameter(torch.tensor([1.0])))
        self.register_parameter("a_coeffs", torch.nn.Parameter(torch.randn(n_freqs)))
        self.register_parameter("b_coeffs", torch.nn.Parameter(torch.randn(n_freqs)))
        self.register_parameter("w", torch.nn.Parameter(torch.tensor([1.0])))

    def get_warped_z(self, x1, x2):
        def get_ell(z):
            fourier_sum = 0
            for k in range(1, self.n_freqs + 1):
                fourier_sum += self.a_coeffs[k - 1] * torch.cos(k * self.w * z)
                fourier_sum += self.b_coeffs[k - 1] * torch.sin(k * self.w * z)
            return torch.abs(self.c + fourier_sum) + 1e-4

        z1_warped = x1 / get_ell(x1)
        z2_warped = x2 / get_ell(x2)
        return z1_warped, z2_warped

    def forward(self, x1, x2, diag=False, **params):
        z1_warped, z2_warped = self.get_warped_z(x1, x2)
        return self.base_kernel(z1_warped, z2_warped, diag=diag, **params)