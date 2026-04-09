
import torch
from nak_torch.algorithms import msip, msip_gs
from nak_torch.algorithms.msip import MSIPGMMGaussianKernel
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
    precisions = [5.0, 5.0, 2.0]
    weights    = [1/3, 1/3, 1/3]
    log_probs = []
    for mean, prec, w in zip(means, precisions, weights):
        diff = pt - mean
        lp = torch.log(torch.tensor(w)) - 0.5 * prec * torch.linalg.norm(diff, dim=-1)**2
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()

n_steps, n_particles = 5000, 20
lr_msip = 5e-3
kernel_length_scale = 0.9
kernel_diag_infl = 1e-8
bounds = (-100., 100.)

init_mean = torch.tensor([18.0, 18.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))

gmm_weights = torch.tensor([1/3, 1/3, 1/3])
gmm_means = torch.stack([
    torch.tensor([6.2,  -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([7.0,   0.0]),
])
gmm_covs = torch.stack([torch.eye(2) / 5.0] * 3)

estimator = MSIPGMMGaussianKernel(
    weights=gmm_weights,
    means=gmm_means,
    covariances=gmm_covs,
    bandwidth=kernel_length_scale
)

trajectories_msip, traj_wts_msip = msip(
    estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)

trajectories_msip_gs, traj_wts_msip_gs = msip_gs(
    estimator, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True
)

def gmm_rbf_expectations(particles, weights, means, covs, bandwidth):
    sigma_sq = bandwidth ** 2
    K, D = means.shape
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for l in range(K):
            cov_jl = covs[j] + covs[l] + 2 * sigma_sq * eye
            diff_jl = means[j] - means[l]
            log_k = -0.5 * (diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                            + torch.logdet(2 * torch.pi * cov_jl))
            Epp = Epp + weights[j] * weights[l] * log_k.exp()
    smoothed_covs = covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - means[k].unsqueeze(0)
        log_k = -0.5 * (
            torch.einsum("ni,ij,nj->n", diff_nk, torch.linalg.inv(smoothed_covs[k]), diff_nk)
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        )
        Exp = Exp + weights[k] * log_k.exp().mean()
    return Epp, Exp

mmd_every = 10
steps_recorded = list(range(0, n_steps, mmd_every))
mmd_msip, mmd_msip_gs = [], []

for i in steps_recorded:
    for traj, mmd_list in [
        (trajectories_msip,    mmd_msip),
        (trajectories_msip_gs, mmd_msip_gs),
    ]:
        pts = traj[i]
        Epp, Exp = gmm_rbf_expectations(pts, gmm_weights, gmm_means, gmm_covs, kernel_length_scale)
        Kxx = torch.exp(-torch.cdist(pts, pts).pow(2) / (2 * kernel_length_scale ** 2)).mean()
        mmd_val = (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()
        mmd_list.append(mmd_val)

Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

pts_msip    = trajectories_msip[-1]
wts_msip    = traj_wts_msip[-1];    wts_msip    = wts_msip    / wts_msip.sum()
pts_msip_gs = trajectories_msip_gs[-1]
wts_msip_gs = traj_wts_msip_gs[-1]; wts_msip_gs = wts_msip_gs / wts_msip_gs.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contour(X, Y, Z, levels=20)
axes[0].scatter(pts_msip[:, 0],    pts_msip[:, 1],    c=wts_msip,    alpha=0.5)
axes[0].set_title("MSIP")
axes[0].set_aspect(1.0)
axes[1].contour(X, Y, Z, levels=20)
axes[1].scatter(pts_msip_gs[:, 0], pts_msip_gs[:, 1], c=wts_msip_gs, alpha=0.5)
axes[1].set_title("MSIP-GS")
axes[1].set_aspect(1.0)
plt.tight_layout()
plt.show()


Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

pts_msip    = trajectories_msip[-1]
wts_msip    = traj_wts_msip[-1];    wts_msip    = wts_msip    / wts_msip.sum()
pts_msip_gs = trajectories_msip_gs[-1]
wts_msip_gs = traj_wts_msip_gs[-1]; wts_msip_gs = wts_msip_gs / wts_msip_gs.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].contour(X, Y, Z, levels=20)
axes[0].scatter(pts_msip[:, 0],    pts_msip[:, 1],    c=wts_msip,    alpha=0.5)
axes[0].set_title("MSIP")
axes[0].set_xlim(-10, 15)
axes[0].set_ylim(-10, 15)
axes[0].set_aspect(1.0)
axes[1].contour(X, Y, Z, levels=20)
axes[1].scatter(pts_msip_gs[:, 0], pts_msip_gs[:, 1], c=wts_msip_gs, alpha=0.5)
axes[1].set_title("MSIP-GS")
axes[1].set_xlim(-10, 15)
axes[1].set_ylim(-10, 15)
axes[1].set_aspect(1.0)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps_recorded, mmd_msip,    label="MSIP")
ax.plot(steps_recorded, mmd_msip_gs, label="MSIP-GS")
ax.set_xlabel("Step")
ax.set_ylabel("MMD to target GMM")
ax.legend()
plt.tight_layout()
plt.show()




