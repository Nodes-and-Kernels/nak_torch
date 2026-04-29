# %%
import nest_asyncio
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import nak_torch
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm
from nak_torch.tools import stan_tools
from nak_torch.tools.types import BatchGradLogDensityEvaluator

nest_asyncio.apply() # See pystan documentation on why you need this when doing jupyter
import stan  # noqa: E402
from posteriordb import PosteriorDatabaseGithub # noqa: E402

# %%
if "GITHUB_PAT" not in os.environ.keys():
    raise ValueError("Expected GITHUB_PAT to be in environment. Please add this into, e.g., your .env file.")

my_pdb = PosteriorDatabaseGithub()
pos = my_pdb.posterior_names()

def sample_tau_prior(N_samples, loc: float = 0., scale: float = 5.):
    dist = torch.distributions.Cauchy(loc, scale, True)
    return dist.rsample((N_samples,)).abs_()

# %%
posterior = my_pdb.posterior("eight_schools-eight_schools_centered")

# %%
post_model = stan.build(posterior.model.stan_code(), data=posterior.data.values())

# %%
stan_model = stan_tools.StanModel(post_model)

# %%
pts = torch.randn((100, stan_model.dim))
pdfs = stan_model.log_dens_batch(pts, None)
grad_log_pdfs = stan_model.grad_log_dens_batch(pts, None)
grad_log_pdfs_2, pdfs_2 = stan_model.grad_val_log_dens_batch(pts, None)

# %%
GRADIENT_DECAY = 1.0
N_PARTICLES = 100
KERNEL_DIAG_INFL = 1e-6
KERNEL_LENGTHSCALE = 1e-1
BOUNDS = (-100.0, 100.0)
target_msip_fr = MSIPFredholm(GRADIENT_DECAY, stan_model.grad_val_log_dens_batch)
init_eta = torch.randn((N_PARTICLES, 8))
init_tau = sample_tau_prior(N_PARTICLES).clamp_(*BOUNDS)
init_mu = torch.randn((N_PARTICLES, 1)) * 5
init_particles = torch.column_stack((init_mu, init_tau, init_eta))

msip = MSIP(
    stan_model.dim,
    N_PARTICLES,
    kernel_diag_infl=KERNEL_DIAG_INFL,
    kernel_lengthscale=KERNEL_LENGTHSCALE,
)

# %%
N_STEPS = 1000
LR = 1e-3
trajectories_msip_fr = nak_torch.nak(
    target_msip_fr,
    msip,
    N_STEPS,
    LR,
    init_particles=init_particles,
    bounds=BOUNDS,
)
trajectories_pts_msip_fr, trajectories_wts_msip_fr = trajectories_msip_fr

# %%
target_svgd = BatchGradLogDensityEvaluator(
    stan_model.grad_log_dens_batch,
    is_grad=True,
    is_batched=True
)

svgd = SVGD(
    stan_model.dim,
    N_PARTICLES,
    kernel_lengthscale=KERNEL_LENGTHSCALE,
    kernel_lengthscale_quantile=0.5
)

# %%
N_STEPS = 1000
LR = 1e-3
trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    N_STEPS,
    LR,
    init_particles=init_particles,
    bounds=BOUNDS,
)


# %%
draws = stan_tools.get_draws(post_model, posterior)

# %%
cross_ent = nak_torch.metrics.CrossEntropy(stan_model.log_dens_batch)

# %%
msip_cross_ent = [cross_ent(pts, None, None) for pts in tqdm(trajectories_pts_msip_fr)]
svgd_cross_ent = [cross_ent(pts, None, None) for pts in tqdm(trajectories_pts_svgd)]

# %%
plt.plot(msip_cross_ent, label="MSIP")
plt.plot(svgd_cross_ent, label="SVGD")
plt.plot()
plt.title("Cross entropy across iterations")
plt.legend()
# %%
