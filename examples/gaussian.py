# Gaussian example with all algorithms.
# %%
from functools import partial
import math

from jaxtyping import Float
import matplotlib.pyplot as plt
import torch
from torch import Tensor

import nak_torch
from nak_torch.algorithms import SVGD, MSIP, MSIPGS, GradALDI, CBS, GradFreeALDI, EKS
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
from nak_torch.tools.types import BatchLogDensityGradEvaluator, BatchLogDensityEvaluator

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# %%
def make_gaussian_post(
    forward_op: Float[Tensor, "obs dim"],
    mean_pr: Float[Tensor, " dim"],
    cov_pr: Float[Tensor, "dim dim"],
    mean_li: Float[Tensor, " obs"],
    cov_li: Float[Tensor, "obs obs"],
):
    forward_op = forward_op.T
    cov_post = torch.linalg.inv(
        forward_op.T @ torch.linalg.solve(cov_li, forward_op) + torch.linalg.inv(cov_pr)
    )
    mean_post = cov_post @ (
        forward_op.T @ torch.linalg.solve(cov_li, mean_li)
        + torch.linalg.solve(cov_pr, mean_pr)
    )
    return mean_post, cov_post


def weighted_cov(pts: Tensor, wts: Tensor):
    mean = wts @ pts
    second_moment = torch.einsum("b,bi,bj", wts, pts, pts)
    return second_moment - mean.outer(mean)


# %%
torch.manual_seed(1023921)
obs_op = torch.randn(2, 5)
obs_op.div_(obs_op.norm(dim=1, keepdim=True))
forward_model = torch.compile(lambda particles, _=None: particles @ obs_op)
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) + 20

model = nak_torch.GaussianModel(
    forward_model,
    likelihood_precision=10.0,
    prior_precision=0.9,
    true_obs=true_obs,
    is_vectorized=True,
)

post_log_dens = model.to_log_dens()


def _tmp_post_log_dens(pts, args):
    out = post_log_dens(pts, args)
    return out.sum(), out


post_log_dens_grad_val_batch = torch.func.grad(_tmp_post_log_dens, has_aux=True)

# %%
mean_pr, cov_pr = torch.zeros(2), torch.eye(2) / model.prior_precision
mean_li, cov_li = (
    model.true_obs,
    torch.eye(len(model.true_obs)) / model.likelihood_precision,
)

mean_post, cov_post = make_gaussian_post(obs_op, mean_pr, cov_pr, mean_li, cov_li)
vals, vecs = torch.linalg.eigh(cov_post)
cov_post_sqrt = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T
samps = torch.randn(10000, 2) @ cov_post_sqrt + mean_post

# %%
n_steps, n_particles = 1000, 500
lr = 1e-2
bounds = (-100.0, 100.0)
rng = torch.Generator(device=torch.get_default_device())
rng.manual_seed(0)

init_particles = (
    torch.randn((n_particles, 2), generator=rng) / model.prior_precision
    + model.prior_mean
)


# %%
eks = EKS(dim=2, n_particles=n_particles, rng_or_seed=rng)
trajectories_eks = nak_torch.nak(
    model,
    eks,
    n_steps=n_steps,
    lr=lr,
    rng_or_seed=rng,
    target_args=None,
    bounds=bounds,
    init_particles=init_particles,
)

# %%
grad_aldi = GradALDI(dim=2, n_particles=n_particles, rng=rng)
grad_aldi_target = BatchLogDensityGradEvaluator(
    post_log_dens, is_grad=False, is_batched=True
)
trajectories_galdi = nak_torch.nak(
    grad_aldi_target,
    grad_aldi,
    n_steps=n_steps,
    lr=lr,
    init_particles=init_particles,
    keep_all=False,
    rng_or_seed=rng,
    target_args=None,
    bounds=bounds,
)

# %%
gf_aldi = GradFreeALDI(dim=2, n_particles=n_particles)
trajectories_gfaldi = nak_torch.nak(
    model,
    gf_aldi,
    n_steps=n_steps,
    lr=1e-2,
    init_particles=init_particles,
    keep_all=True,
    rng_or_seed=rng,
    target_args=None,
    bounds=bounds,
)

# %%
cbs_target = BatchLogDensityEvaluator(post_log_dens, is_batched=True)
cbs = CBS(dim=2, n_particles=n_particles, default_inverse_temp=0.5, rng=rng)
trajectories_cbs = nak_torch.nak(
    cbs_target,
    cbs,
    5000,
    lr=lr,
    rng_or_seed=rng,
    init_particles=init_particles,
    target_args=None,
    bounds=bounds,
)

# %%
target_svgd = BatchLogDensityGradEvaluator(
    post_log_dens, is_grad=False, is_batched=True
)
svgd = SVGD(
    dim=2,
    n_particles=n_particles,
    kernel_lengthscale_quantile=0.5,  # Median heuristic
)
trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    n_steps=n_steps,
    lr=lr,
    init_particles=init_particles,
    bounds=bounds,
    target_args=None,
)

# %%
kernel_lengthscale = 0.1
gradient_decay = 0.95
n_particles_msip = 25
n_steps_msip = 1000
lr_msip = 0.1
kernel_diag_infl = 1e-6
msip = MSIP(
    dim=2,
    n_particles=n_particles_msip,
    kernel_diag_infl=kernel_diag_infl,
    kernel_lengthscale=kernel_lengthscale,
    # kernel_lengthscale_quantile=0.25 # If you want adaptive bandwidth.
)

# %%
msip_fredholm_target = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)

trajectories_pts_msip_fr, trajectories_wts_msip_fr = nak_torch.nak(
    msip_fredholm_target,
    msip,
    n_steps_msip,
    lr_msip,
    rng_or_seed=rng,
    init_particles=init_particles[: msip.n_particles],
    target_args=None,
    keep_all=True,
    bounds=bounds,
)

# %%
msipgs = MSIPGS(
    dim=2,
    n_particles=n_particles_msip,
    kernel_diag_infl=kernel_diag_infl,
    kernel_lengthscale=kernel_lengthscale,
    # kernel_lengthscale_quantile=0.25 # If you want adaptive bandwidth.
)

trajectories_pts_msipgs_fr, trajectories_wts_msipgs_fr = nak_torch.nak(
    msip_fredholm_target,
    msipgs,
    n_steps=500,
    lr=1e-1,
    rng_or_seed=rng,
    init_particles=init_particles[: msipgs.n_particles],
    target_args=None,
    keep_all=True,
    bounds=bounds,
)


# %%
def mc_quad_rule(batch_size: int, N_quad: int = 5, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim), generator=rng)
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad(batch_size: int, N_spherical: int = 5, N_radial: int = 3):
    pts, wts = spherical_MC_radial_Laguerre(batch_size, N_spherical, 2, N_radial)
    return pts, wts


# %%
msip_quadgrad_target = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    # partial(spherical_quad, N_spherical=10, N_radial=4),
    mc_quad_rule,
    1.0
)

trajectories_pts_msip_qg, trajectories_wts_msip_qg = nak_torch.nak(
    msip_quadgrad_target,
    msip,
    2000,
    lr=5e-2,
    rng_or_seed=rng,
    init_particles=init_particles[: msip.n_particles],
    target_args=None,
    keep_all=False,
    bounds=bounds,
)

# %%
msip_quadgf_target = MSIPQuadGradientFree(
    post_log_dens, partial(spherical_quad, N_spherical=10, N_radial=3)
)

trajectories_pts_msip_qgf, trajectories_wts_msip_qgf = nak_torch.nak(
    msip_quadgf_target,
    msip,
    n_steps=n_steps_msip,
    lr=5e-2,
    rng_or_seed=rng,
    init_particles=init_particles[: msip.n_particles],
    target_args=None,
    keep_all=False,
    bounds=bounds,
)


# %%
pts_eks = trajectories_eks[-1]
pts_galdi = trajectories_galdi[-1]
pts_gfaldi = trajectories_gfaldi[-1]
pts_cbs = trajectories_cbs[-1]
idx_msip = -1
alpha_msip = 2 / math.sqrt(n_particles_msip)
pts_msip_fr = trajectories_pts_msip_fr[idx_msip]
wts_msip_fr = trajectories_wts_msip_fr[idx_msip]
idx_msip_gs = -1
pts_msipgs_fr = trajectories_pts_msipgs_fr[idx_msip_gs]
wts_msipgs_fr = trajectories_wts_msipgs_fr[idx_msip_gs]
pts_msip_qg = trajectories_pts_msip_qg[-1]
wts_msip_qg = trajectories_wts_msip_qg[-1]
pts_msip_qgf = trajectories_pts_msip_qgf[-1]
wts_msip_qgf = trajectories_wts_msip_qgf[-1]
pts_svgd = trajectories_pts_svgd[-1]


Ngrid = 100
xgrid = torch.linspace(-1, 1, Ngrid)
xgrid = 3 * xgrid * cov_post_sqrt[0, 0] + mean_post[0]
ygrid = torch.linspace(-1, 1, Ngrid)
ygrid = 3 * ygrid * cov_post_sqrt[1, 1] + mean_post[1]
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)

fig, ax = plt.subplots()
ax.contour(X, Y, post_log_dens(grid_pts, model).reshape(Ngrid, Ngrid), levels=10)
# ax.scatter(samps[:, 0], samps[:, 1], alpha=0.025, label="Truth")
# ax.scatter(pts_galdi[:, 0], pts_galdi[:, 1], alpha=0.2, label="Grad-ALDI")
# ax.scatter(pts_gfaldi[:, 0], pts_gfaldi[:, 1], alpha=0.2, label="GradFree-ALDI")
# ax.scatter(pts_eks[:, 0], pts_eks[:, 1], alpha=0.1, label="EKS")
# ax.scatter(pts_cbs[:, 0], pts_cbs[:, 1], alpha=0.1, label="CBS")
# ax.scatter(pts_svgd[:, 0], pts_svgd[:, 1], alpha=0.1, label="SVGD")
# ax.scatter(pts_msip_fr[:, 0], pts_msip_fr[:, 1], alpha=alpha_msip, label="MSIP")
# ax.scatter(pts_msipgs_fr[:, 0], pts_msipgs_fr[:, 1], alpha=alpha_msip, label="MSIP-GS")
ax.scatter(pts_msip_qg[:, 0], pts_msip_qg[:, 1], alpha=alpha_msip, label="MSIP-QuadGrad")
# ax.scatter(pts_msip_qgf[:, 0], pts_msip_qgf[:, 1],
#                s = 50*wts_msip_qgf.abs()/wts_msip_qgf.max(), alpha=alpha_msip, label="MSIP-QuadGradFree")
# plt.colorbar(s)
# ax.set_aspect(1.0)
ax.legend()
# ax.set_xlim(xgrid.min(), xgrid.max())
# ax.set_ylim(ygrid.min(), ygrid.max())
plt.show()

# %%

# %%
print(f"""
Covariances---
Truth:
{cov_post}
EKS:
{pts_eks.T.cov()}
Grad-ALDI:
{pts_galdi.T.cov()}
GradFree-ALDI:
{pts_gfaldi.T.cov()}
MSIP:
{weighted_cov(pts_msip, wts_msip)}
MSIP-QuadGrad:
{weighted_cov(pts_msip_qg, wts_msip_qg)}
MSIP-QuadGradFree:
{weighted_cov(pts_msip_qgf, wts_msip_qgf)}
""")

# %%
