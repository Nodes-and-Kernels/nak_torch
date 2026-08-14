# %%
from typing import Optional
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from jaxtyping import Float
from torch import Tensor

from pyro.infer import mcmc

def create_pyro_gmm(
        means: Optional[Float[Tensor, "components dim"]] = None,
        precs: Optional[Float[Tensor, "components dim"]] = None,
        weights: Optional[Float[Tensor, " components"]] = None,
        dim: Optional[int] = None
):
    n_components: int = 0
    if means is None:
        if precs is None:
            if dim is None:
                raise ValueError("Expected dim to be an integer when means, covs, weights are None")
        else:
            dim = precs.shape[1]
            if dim != precs.shape[2]:
                raise ValueError(f"Expected last two dims of covs equal. Got {precs.shape}")
            n_components = precs.shape[0]
    else:
        if precs is None:
            n_components, dim = means.shape
            precs = torch.eye(dim).unsqueeze_(0).repeat((n_components, 1, 1))
    if weights is None:
        weights = torch.ones(n_components)/n_components
    assert precs is not None and means is not None and weights is not None
    def ppl_gmm():
        mixture_component = pyro.sample("mixture_component", dist.Categorical(probs = weights, validate_args=True))
        mean, prec = means[mixture_component], precs[mixture_component]
        return pyro.sample("theta", dist.MultivariateNormal(mean, precision_matrix=prec, validate_args=True))

    return ppl_gmm

# %%
gmm_weights = torch.tensor([1 / 5] * 5)

gmm_means = torch.stack([
    torch.tensor([6.2, -6.0]),
    torch.tensor([-4.0, 5.0]),
    torch.tensor([7.0, 3.0]),
    torch.tensor([-6.5, -4.5]),
    torch.tensor([1.0, 7.0]),
])

gmm_covs = torch.stack([
    0.5 * torch.tensor([[1.5, 0.1], [0.1, 0.5]]),
    0.5 * torch.tensor([[2.0, -0.6], [-0.6, 0.5]]),
    0.5 * torch.tensor([[0.7, 0.4], [0.4, 1.2]]),
    0.5 * torch.tensor([[1.3, -0.5], [-0.5, 0.9]]),
    0.5 * torch.tensor([[0.6, 0.35], [0.35, 1.6]]),
])
gmm_precs = torch.linalg.inv(gmm_covs)

# %%
USE_COMPILE = True
pyro_gmm = create_pyro_gmm(gmm_means, gmm_precs, gmm_weights)
hmc_kernel = mcmc.NUTS(pyro_gmm, jit_compile=USE_COMPILE)

# %%
N_STEPS_PER_CHAIN = 1000
N_STEPS_WARMUP = 100
N_CHAINS = 100
INIT_MEAN = torch.tensor([8.0, 8.0])
INIT_STD = 1.0

initial_params = {
    "theta": torch.randn((N_CHAINS,2)).mul_(INIT_STD).add_(INIT_MEAN)
}
hmc_setup = mcmc.MCMC(
    hmc_kernel,
    num_samples=N_STEPS_PER_CHAIN,
    warmup_steps=N_STEPS_WARMUP,
    initial_params=initial_params,
    num_chains=N_CHAINS
)
hmc_setup.run()

# %%
N_THETA_SKIP = 50
hmc_samples = hmc_setup.get_samples()
hmc_theta = hmc_samples["theta"]
thinned_hmc_samples = hmc_theta[::N_THETA_SKIP]

# %%
def log_pdf(pt, means: Tensor, precs: Tensor, weights: Tensor):
    assert pt.shape == (means.shape[1],)
    diffs = means - pt.unsqueeze(0)
    quad_form = torch.einsum("ki,kij,kj->k", diffs, precs, diffs)
    logdets = torch.logdet(precs)
    component_logpdfs = quad_form.add_(logdets).mul_(-0.5).add_(weights.log())
    return torch.logsumexp(component_logpdfs, dim=0)

log_pdf_v = torch.vmap(log_pdf, in_dims=(0, None, None, None))

# %%
N_steps = 1000
xgrid = torch.linspace(-12., 12., steps=N_steps)
ygrid = torch.linspace(-12., 12., steps=N_steps)
xmesh, ymesh = torch.meshgrid(xgrid, ygrid, indexing="ij")
pt_mesh = torch.column_stack((xmesh.flatten(), ymesh.flatten()))
log_pdf_vals = log_pdf_v(pt_mesh, gmm_means, gmm_precs, gmm_weights).reshape(N_steps, N_steps)

# %%
fig, ax = plt.subplots()
ax.contour(xgrid, ygrid, log_pdf_vals, levels=50)
ax.scatter(thinned_hmc_samples[:,0], thinned_hmc_samples[:,1])
plt.show()

# %%
