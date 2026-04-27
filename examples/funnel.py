# %%
import torch
import matplotlib.pyplot as plt
import nak_torch
from viz_tools import animate_trajectories_box
from functions import funnel
from nak_torch.algorithms import msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
from datetime import datetime
from nak_torch.tools.kernel import kernel_optimal_weight_factory, DEFAULT_KERNEL_MATRIX
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(19230182)

# %%
DIM = 2
funnel_sampler = funnel.sample_factory(DIM)
rng = torch.Generator()
rng.manual_seed(0)
samples = funnel_sampler(rng, 10000)

# %%
N_PARTICLES = 50

init_particles = torch.randn((N_PARTICLES, DIM))

params = {
    "bounds": (-1000, 1000),
    "kernel_length_scale": 0.05,
    "init_particles": init_particles,
    "n_particles": N_PARTICLES,
    "dim": 2,
    "lr": 0.01,
    "kernel_diag_infl": 1e-4,
}

# %%
funnel_log_density = funnel.logpdf_factory(DIM)
estimator_fredholm = MSIPFredholm(
    gradient_decay=0.95,
    log_dens_grad_val=torch.vmap(torch.func.grad_and_value(funnel_log_density))
)

trajectories_fr, trajectories_wts_fr = msip(
    estimator_fredholm,
    n_steps=5000,
    use_quantile_length_scale=0.0,
    verbose=True,
    seed=1,
    **params
)
trajectories_fr[-1]

# %%
msip_pts = trajectories_fr[-1]
msip_wts = trajectories_wts_fr[-1]
plt.scatter(samples[:,0], samples[:,1], alpha=0.15)
plt.scatter(msip_pts[:,0], msip_pts[:,1], s = 50*msip_wts/msip_wts.max())
plt.show()