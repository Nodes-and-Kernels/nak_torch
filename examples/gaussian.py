# %%
from functools import partial

from jaxtyping import Float
import matplotlib.pyplot as plt
import torch
from torch import Tensor

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, kfrflow
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientInformed, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre

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
    cov_li: Float[Tensor, "obs obs"]
):
    forward_op = forward_op.T
    cov_post = torch.linalg.inv(
        forward_op.T @ torch.linalg.solve(
            cov_li, forward_op
        ) + torch.linalg.inv(cov_pr)
    )
    mean_post = cov_post @ (forward_op.T @ torch.linalg.solve(
        cov_li, mean_li
    ) + torch.linalg.solve(cov_pr, mean_pr))
    return mean_post, cov_post


def weighted_cov(pts: Tensor, wts: Tensor):
    mean = wts @ pts
    second_moment = torch.einsum("b,bi,bj", wts, pts, pts)
    return second_moment - mean.outer(mean)


# %%
torch.manual_seed(1023921)
obs_op = torch.randn(2, 5)
obs_op.div_(obs_op.norm(dim=1, keepdim=True))
forward_model = torch.compile(lambda particles: particles @ obs_op)
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) + 20

model = nak_torch.GaussianModel(
    forward_model, likelihood_precision=10.0,
    prior_precision=0.9, true_obs=true_obs,
    is_vectorized=True
)


@torch.compile
def like_log_dens(pt):
    ll_term = model.likelihood_precision * \
        torch.linalg.norm(pt @ obs_op - model.true_obs, dim=-1)**2
    return -0.5 * ll_term.squeeze()


@torch.compile
def post_log_dens(pt):
    ll_term = model.likelihood_precision * \
        torch.linalg.norm(pt @ obs_op - model.true_obs, dim=-1)**2
    prior_term = model.prior_precision * torch.linalg.norm(pt, dim=-1)**2
    return -0.5 * (ll_term + prior_term).squeeze()


post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# %%
mean_pr, cov_pr = torch.zeros(2), torch.eye(2) / model.prior_precision
mean_li, cov_li = model.true_obs, torch.eye(
    len(model.true_obs)
) / model.likelihood_precision

mean_post, cov_post = make_gaussian_post(
    obs_op, mean_pr, cov_pr, mean_li, cov_li
)
vals, vecs = torch.linalg.eigh(cov_post)
cov_post_sqrt = vecs @ torch.diag(torch.sqrt(vals)) @ vecs.T
samps = torch.randn(10000, 2) @ cov_post_sqrt + mean_post

# %%
n_steps, n_particles = 1000, 500
lr = 0.1
init_particles = torch.randn((n_particles, 2)) / \
    model.prior_precision + model.prior_mean


# %%
trajectories_eks = eks(
    model, n_particles=n_particles,
    n_steps=n_steps, dim=2, lr=lr,
    init_particles=init_particles, keep_all=False,
)

# %%
# init_particles = torch.randn((n_particles, 2)) + torch.tensor([3, -3])
# delta_ts = torch.ones(1000)/1000
n_particles_kfr = 100
init_kfr = init_particles[:n_particles_kfr] #torch.randn((n_particles_kfr,2)) + torch.tensor([3,-5])
def imq(pt1,pt2,h):
    return 1/torch.sqrt(1 + (torch.linalg.norm(pt1-pt2) / h)**2)
trajectories_kfr = kfrflow(
    like_log_dens,
    n_particles_kfr,
    10000, 2,
    init_particles=init_kfr,
    kernel_length_scale = 1e-2,
    kernel_diag_infl=1e-5,
    # bounds=(-10,10),
    # kernel_elem=imq,
    keep_all=False
)

# %%
from nak_torch.tools.kernel import sqexp_kernel_elem as kernel_elem, sqexp_kernel_matrix
from tqdm import tqdm
kernel_vec = torch.compile(torch.vmap(kernel_elem, in_dims=(None,0,None)))
jac_kernel_vec = torch.vmap(torch.func.grad(kernel_elem), in_dims = (None, 0, None))
kernel_mat = sqexp_kernel_matrix
n_steps_kfr = 100
delta_t = 1 / n_steps_kfr
particles = init_particles.clone()
kernel_length_scale = 1e-2
grad_ks = torch.empty((n_particles, n_particles, 2))
M_t = torch.empty((n_particles, n_particles))
for n in tqdm(range(n_steps_kfr)):
    log_likely_evals = like_log_dens(particles)
    M_t.zero_()
    for i in range(n_particles):
        grad_K = jac_kernel_vec(particles[i], particles, kernel_length_scale)
        grad_ks[i].copy_(grad_K)
        M_t.add_(grad_K @ grad_K.T)
    M_t = M_t.div_(n_particles)
    M_t[torch.arange(n_particles), torch.arange(n_particles)] += 1e-4
    wts_shift = log_likely_evals.mean()
    wts = log_likely_evals.sub_(wts_shift).div_(n_particles)
    K_mat = kernel_mat(particles, kernel_length_scale)
    kernelized_wts = K_mat @ wts
    particles += torch.einsum("jid,i->jd", grad_ks, torch.linalg.solve(M_t, kernelized_wts)).mul_(delta_t)


# %%
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=False
)

# %%
trajectories_gfaldi = gradfree_aldi(
    model, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True
)

# %%
trajectories_cbs = cbs(
    post_log_dens, n_particles, n_steps, inverse_temp=0.95, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True
)

# %%
kernel_length_scale = 0.25
bounds = (-100., 100.)
gradient_decay = 1.0
n_particles_msip = 5
n_steps_msip = 1000
lr_msip = 1e-2
kernel_diag_infl = 1e-8
msip_fredholm = MSIPFredholm(
    gradient_decay,
    post_log_dens_grad_val_batch
)

trajectories_msip, traj_wts_msip = msip(
    msip_fredholm, n_particles_msip, n_steps_msip, dim=2,
    lr=lr_msip, init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    gradient_decay=gradient_decay,
    keep_all=True
)

# %%


def mc_quad_rule(batch_size: int, N_quad: int = 5, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad(batch_size: int, N_spherical: int = 5, N_radial: int = 3):
    pts, wts = spherical_MC_radial_Laguerre(
        batch_size, N_spherical, 2, N_radial)
    return pts, wts


# %%
# kernel_length_scale = 1e-3
# gradient_decay = 1.
msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch, mc_quad_rule,
    gradient_decay
)

trajectories_msip_qg, traj_wts_msip_qg = msip(
    msip_quadgrad, n_particles_msip, 100, dim=2,
    lr=10., init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    # is_log_density_batched=True,
    kernel_diag_infl=1e-8,
    bounds=(-1000, 1000),
    # gradient_decay=gradient_decay,
    keep_all=False
)

# %%
# n_particles_msip = 500
# kernel_length_scale = 1e-2
msip_quadgf = MSIPQuadGradientFree(
    post_log_dens_batch, partial(mc_quad_rule, N_quad=100)
)

trajectories_msip_qgf, traj_wts_msip_qgf = msip(
    msip_quadgf, n_particles_msip, 500, dim=2,
    lr=1., init_particles=init_particles[:n_particles_msip],
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=1e-8,
    bounds=(-1000., 1000.),
    keep_all=False
)


# %%
pts_eks = trajectories_eks[-1]
pts_kfr = particles
pts_galdi = trajectories_galdi[-1]
pts_gfaldi = trajectories_gfaldi[-1]
pts_cbs = trajectories_cbs[-1]
idx_msip = 100
pts_msip = trajectories_msip[idx_msip]
wts_msip = traj_wts_msip[idx_msip]
# wts_msip /= wts_msip.sum()
pts_msip_qg = trajectories_msip_qg[-1]
wts_msip_qg = traj_wts_msip_qg[-1]
wts_msip_qg = wts_msip_qg/wts_msip_qg.sum()
pts_msip_qgf = trajectories_msip_qgf[-1]
wts_msip_qgf = traj_wts_msip_qgf[-1]
# wts_msip_qgf = wts_msip_qgf/wts_msip_qgf.sum()

Ngrid = 100
xgrid = torch.linspace(-1, 1, Ngrid)
xgrid = 3 * xgrid * cov_post_sqrt[0, 0] + mean_post[0]
ygrid = torch.linspace(-1, 1, Ngrid)
ygrid = 3 * ygrid * cov_post_sqrt[1, 1] + mean_post[1]
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)

fig, ax = plt.subplots()
ax.contour(X, Y, post_log_dens(grid_pts).reshape(Ngrid, Ngrid), levels=10)
# ax.scatter(samps[:, 0], samps[:, 1], alpha=0.025, label="Truth")
# ax.scatter(pts_galdi[:, 0], pts_galdi[:, 1], alpha=0.2, label="Grad-ALDI")
# ax.scatter(pts_gfaldi[:, 0], pts_gfaldi[:, 1],
#            alpha=0.2, label="GradFree-ALDI")
ax.scatter(pts_kfr[:,0], pts_kfr[:,1], label="KFR")
# ax.scatter(pts_eks[:, 0], pts_eks[:, 1], alpha=0.1, label="EKS")
# ax.scatter(pts_cbs[:, 0], pts_cbs[:, 1], alpha=0.1, label="CBS")
# s = ax.scatter(pts_msip[:, 0], pts_msip[:, 1],
            #    c=wts_msip, alpha=0.15, label="MSIP")
# s = ax.scatter(pts_msip_qg[:, 0], pts_msip_qg[:, 1],
#                c = wts_msip_qg, alpha=0.15, label="MSIP-QuadGrad")
# s = ax.scatter(pts_msip_qgf[:, 0], pts_msip_qgf[:, 1],
#                c = wts_msip_qgf, alpha=0.15, label="MSIP-QuadGradFree")
# plt.colorbar(s)
ax.set_aspect(1.0)
ax.legend()
plt.show()

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
