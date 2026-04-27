# %%
import nest_asyncio
import torch
import nak_torch
from nak_torch.tools import stan_tools
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientFree

nest_asyncio.apply() # See pystan documentation on why you need this when doing jupyter
import stan  # noqa: E402

# %%
# Example from https://github.com/stan-dev/pystan
schools_code = """
data {
  int<lower=0> J;         // number of schools
  array[J] real y;              // estimated treatment effects
  array[J] real<lower=0> sigma; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real log_tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + exp(log_tau) * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(log_tau | 5, 1);
  target += normal_lpdf(mu | 0, 10);
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""

schools_data = {
    "J": 8,
    "y": [28, 8, -3, 7, -1, 1, 18, 12],
    "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
}

posterior = stan.build(schools_code, data=schools_data)

# %%
# Ten dimensional (mu, tau, eta): theta is a constrained parameter.
model = stan_tools.StanModel(posterior, dim=10)

# %%
# Test evaluation of the pdf and logpdf
pts = torch.randn((100, model.dim))
pdfs = model.log_dens_batch(pts, None)
grad_log_pdfs = model.grad_log_dens_batch(pts, None)
grad_log_pdfs_2, pdfs_2 = model.grad_val_log_dens_batch(pts, None)

# %%
GRADIENT_DECAY = 0.95
N_PARTICLES = 100
KERNEL_DIAG_INFL = 1e-6
KERNEL_LENGTHSCALE = 1e-2
target_msip_fr = MSIPFredholm(GRADIENT_DECAY, model.grad_val_log_dens_batch)
init_eta = torch.randn((N_PARTICLES, 8))
init_log_tau = torch.randn((N_PARTICLES, 1)) + 5
init_mu = torch.randn((N_PARTICLES, 1)) * 10
init_particles = torch.column_stack((init_mu, init_log_tau, init_eta))
msip = MSIP(
    model.dim,
    N_PARTICLES,
    kernel_diag_infl=KERNEL_DIAG_INFL,
    kernel_lengthscale=KERNEL_LENGTHSCALE,
    kernel_lengthscale_quantile=0.05
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
    bounds=(-100.0, 100.0),
)
trajectories_pts_msip_fr, trajectories_wts_msip_fr = trajectories_msip_fr

# %%
msip_fr_end = trajectories_pts_msip_fr[-1]
eta_end = msip_fr_end[:,:8] - init_particles[:,:8]
mean_sq_shift = (msip_fr_end - init_particles).square().sum() / init_particles.square().sum()
print(mean_sq_shift)