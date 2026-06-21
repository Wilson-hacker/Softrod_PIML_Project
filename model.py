import gpytorch
import torch
import copy


class EnergyGradientGP(gpytorch.models.ExactGP):
    """
    Learns the gradient of the Hamiltonian w.r.t state variables.
    You will instantiate two of these: one for dE/dp, one for dE/dq.
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type="baseline", hybird=False, n_freqs=10):
        super(EnergyGradientGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  # Physics prior: assume 0 energy gradient initially

        if kernel_type == "baseline":
            from kernels import BaselineRBFKernel
            self.covar_module = BaselineRBFKernel()
        elif kernel_type == "P_only":
            from kernels import MomentumKernel
            self.covar_module = MomentumKernel()
        elif kernel_type == "SD":
            from kernels import StructuralDecoupledKernel
            if hybird:
                self.covar_module = StructuralDecoupledKernel(spatial_kernel_type="gibbs", n_freqs=n_freqs)
            else:
                self.covar_module = StructuralDecoupledKernel()
        elif kernel_type == "gibbs":
            from kernels import GibbsStateKernel
            self.covar_module = GibbsStateKernel(n_freqs=n_freqs)
        else:
            raise ValueError("Unknown kernel type")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(train_x, train_y, kernel_type="baseline", hybird=False, n_freqs=10, training_iter=1000, patience=5, min_delta=1e-3):
    """
    Handles the PyTorch optimization loop for the GP with an embedded early-stopping mechanism.

    Args:
        train_x: Training inputs (p, q tensors)
        train_y: Training targets (integrated spatial derivatives)
        kernel_type: String identifier for the kernel architecture
        training_iter: Maximum optimization steps allowed
        patience: Number of iterations to wait for improvement before killing the loop
        min_delta: Minimum reduction in loss to qualify as an improvement
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = EnergyGradientGP(train_x, train_y, likelihood, kernel_type, hybird=hybird, n_freqs=n_freqs)

    # Enable training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Early stopping tracking variables
    best_loss = float('inf')
    best_model_state = None
    best_likelihood_state = None
    patience_counter = 0

    print(f"Beginning GP training with Early Stopping (Patience: {patience}, Min Delta: {min_delta})")

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # Print status updates every 10 iterations
        if (i + 1) % 20 == 0:
            print(f'Iter {i + 1}/{training_iter} - Loss: {current_loss:.3f}')

        # --- Early Stopping Logic ---
        # Check if the current loss improved significantly past our best recorded loss
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            # Save a deep copy checkpoint of the optimal parameters
            best_model_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            patience_counter = 0  # Reset counter since we found a better state
        else:
            patience_counter += 1

        # Trigger termination if patience threshold is breached
        if patience_counter >= patience:
            print(f'==> Early stopping triggered at iteration {i + 1}. Restoring best loss state: {best_loss:.3f}')
            # Rollback parameters to the best historical checkpoint
            model.load_state_dict(best_model_state)
            likelihood.load_state_dict(best_likelihood_state)
            break

    return model, likelihood