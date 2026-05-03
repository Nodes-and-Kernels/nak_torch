# %%
from typing import Optional

import nest_asyncio
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import nak_torch
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm
from nak_torch.tools import stan_tools
from nak_torch.tools.types import BatchGradLogDensityEvaluator, DeviceLike

nest_asyncio.apply() # See pystan documentation on why you need this when doing jupyter
import stan  # noqa: E402
import posteriordb # noqa: E402

def covid_prior_sample(
    N_samples: int,
    M_y: int = 14,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLike] = None,
    rng: Optional[torch.Generator] = None,
):
    if rng is None:
        rng = torch.default_generator
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.get_default_device()
    tau = (
        torch.empty((N_samples, 1), dtype=dtype, device=device)
        .exponential_(generator=rng)
        .div_(0.03)
    )
    y = (
        torch.empty((N_samples, M_y), dtype=dtype, device=device)
        .exponential_(generator=rng)
        .div_(tau)
    )
    phi = torch.randn((N_samples, 1), generator=rng).mul_(5.0)
    kappa = torch.randn((N_samples, 1), generator=rng).mul_(0.5)
    mu = torch.randn((N_samples, M_y), generator=rng).mul_(kappa).add_(3.28)
    alpha_hier = torch._standard_gamma(
        torch.as_tensor(0.1667, dtype=dtype, device=device).expand(N_samples, 6),
        generator=rng,
    )
    ifr_noise = torch.randn((N_samples, M_y), generator=rng).mul_(0.1).add_(1.0)
    log_tau = tau.log_()
    log_alpha_hier = alpha_hier.log_()
    log_y = y.log_()
    return torch.column_stack((mu, log_alpha_hier, kappa, log_y, phi, log_tau, ifr_noise))

# %%
pdb = posteriordb.PosteriorDatabase()
which_posterior = "ecdc0501-covid19imperial_v3"
posterior = pdb.posterior(which_posterior)
post_model = stan.build(posterior.model.stan_code(), data=posterior.data.values())
dim = sum(posterior.information["dimensions"].values())
stan_model = stan_tools.StanModel(post_model, dim)

# %%
pts = torch.randn((10, stan_model.dim))
pdfs = stan_model.log_dens_batch(pts, None)
grad_log_pdfs = stan_model.grad_log_dens_batch(pts, None)
grad_log_pdfs_2, pdfs_2 = stan_model.grad_val_log_dens_batch(pts, None)
assert (pdfs - pdfs_2).square_().sum() < 1e-10
assert (grad_log_pdfs - grad_log_pdfs_2).square_().sum() < 1e-10

# %%
GRADIENT_DECAY = 1.0
N_PARTICLES = 25
KERNEL_DIAG_INFL = 1e-2
KERNEL_LENGTHSCALE = 1e-1
BOUNDS = (-100.0, 100.0)
target_msip_fr = MSIPFredholm(GRADIENT_DECAY, stan_model.grad_val_log_dens_batch)
init_particles = torch.randn((N_PARTICLES, stan_model.dim))#covid_prior_sample(N_PARTICLES)

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
LR = 1e-4
trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    N_STEPS,
    LR,
    init_particles=init_particles,
    bounds=BOUNDS,
)


# %%
crossent = nak_torch.metrics.CrossEntropy(stan_model.log_dens_batch)
msip_crossent = [crossent(p,w) for p,w in tqdm(zip(*trajectories_msip_fr), total=N_STEPS+1)]
svgd_crossent = [crossent(p) for p in tqdm(trajectories_pts_svgd)]

# %%
ksd = nak_torch.metrics.KernelSteinDiscrepancy(stan_model.grad_log_dens_batch, KERNEL_LENGTHSCALE)
msip_ksd = [ksd(p,w) for p,w in tqdm(zip(*trajectories_msip_fr), total=N_STEPS+1)]
svgd_ksd = [ksd(p) for p in tqdm(trajectories_pts_svgd)]

# %%
plt.plot(msip_crossent, label="msip")
plt.plot(svgd_crossent, label="svgd")
plt.title("Cross Entropy")
plt.legend()
plt.show()

# %%
plt.plot(msip_ksd, label="msip")
plt.plot(svgd_ksd, label="svgd")
plt.legend()
plt.title("KSD")
plt.show()