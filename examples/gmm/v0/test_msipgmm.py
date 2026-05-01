import torch
from functools import partial
from nak_torch.algorithms import msip
from nak_torch.algorithms.msip import MSIPQuadGradientInformed, MSIPGMMGaussianKernel

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)

torch.manual_seed(1023921)

def post_log_dens(pt):
    means = [
        torch.tensor([6.2,  -6.0]),
        torch.tensor([-4.0, 5.0]),
        torch.tensor([7.0,  0.0]),
    ]
    precisions = [5.0, 5.0, 5.0]
    weights    = [1/3, 1/3, 1/3]
    log_probs = []
    for mean, prec, w in zip(means, precisions, weights):
        diff = pt - mean
        lp = torch.log(torch.tensor(w)) - 0.5 * prec * torch.linalg.norm(diff, dim=-1)**2
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()

post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

n_steps, n_particles = 500, 5
lr_msip = 80e-3
kernel_length_scale = 2.8
kernel_diag_infl = 1e-8
gradient_decay = 1.0
bounds = (-100., 100.)

init_particles = torch.randn((n_particles, 2))

def mc_quad_rule(batch_size, N_quad=10000, dim=2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts

msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    partial(mc_quad_rule, N_quad=50),
    gradient_decay
)

trajectories_qg, traj_wts_qg = msip(
    msip_quadgrad, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)

gmm_weights = torch.tensor([1/3, 1/3, 1/3])
gmm_means = torch.stack([
    torch.tensor([6.2,  -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([7.0,   0.0]),
])
gmm_covs = torch.stack([torch.eye(2) / 5.0] * 3)

msip_gmm = MSIPGMMGaussianKernel(
    weights=gmm_weights,
    means=gmm_means,
    covariances=gmm_covs,
    bandwidth=kernel_length_scale
)

trajectories_gmm, traj_wts_gmm = msip(
    msip_gmm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)





Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

pts_qg = trajectories_qg[-1]
wts_qg = traj_wts_qg[-1]
wts_qg = wts_qg / wts_qg.sum()

pts_gmm = trajectories_gmm[-1]
wts_gmm = traj_wts_gmm[-1]
wts_gmm = wts_gmm / wts_gmm.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].contour(X, Y, Z, levels=20)
axes[0].scatter(pts_qg[:, 0], pts_qg[:, 1], c=wts_qg, alpha=0.5)
axes[0].set_title("MSIP-QuadGradientInformed")
axes[0].set_aspect(1.0)

axes[1].contour(X, Y, Z, levels=20)
axes[1].scatter(pts_gmm[:, 0], pts_gmm[:, 1], c=wts_gmm, alpha=0.5)
axes[1].set_title("MSIP-GMM")
axes[1].set_aspect(1.0)

plt.tight_layout()
plt.show()
