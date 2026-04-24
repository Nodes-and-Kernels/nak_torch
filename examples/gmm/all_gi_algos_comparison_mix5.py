from functools import partial

import torch
import matplotlib.pyplot as plt

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPGMMGaussianKernel,
)

# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(1023921)


# ── Target: 5-component GMM ───────────────────────────────────────────────────
gmm_weights = torch.tensor([1/5] * 5)

gmm_means = torch.stack([
    torch.tensor([ 6.2, -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([ 7.0,  3.0]),
    torch.tensor([-6.5, -4.5]),
    torch.tensor([ 1.0,  7.0]),
])

gmm_covs = torch.stack([
    0.5 * torch.tensor([[1.5,  0.1],
                        [0.1,  0.5]]),

    0.5 * torch.tensor([[2.0, -0.6],
                        [-0.6, 0.5]]),

    0.5 * torch.tensor([[0.7,  0.4],
                        [0.4,  1.2]]),

    0.5 * torch.tensor([[1.3, -0.5],
                        [-0.5, 0.9]]),

    0.5 * torch.tensor([[0.6,  0.35],
                        [0.35, 1.6]]),
])

gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_grad_val       = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)


# ── Gaussian model wrapper needed by EKS ─────────────────────────────────────
obs_op = torch.randn(2, 5)
obs_op.div_(obs_op.norm(dim=1, keepdim=True))
forward_model = lambda particles: particles @ obs_op
true_obs = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) + 20

model = nak_torch.GaussianModel(
    forward_model,
    likelihood_precision=10.0,
    prior_precision=0.9,
    true_obs=true_obs,
    is_vectorized=True,
)


# ── Shared hyper-parameters ───────────────────────────────────────────────────
n_steps     = 500
n_particles = 50
lr          = 0.5
lr_msip     = 50e-2

kernel_length_scale = 0.5
kernel_diag_infl    = 1e-8
gradient_decay      = 1.0
bounds              = (-100.0, 100.0)

# init_particles = torch.randn((n_particles, 2)) / \
#     model.prior_precision + torch.tensor([3.2, -5.0])
    
    
init_mean = torch.tensor([8.0, 8.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))



# ── Quadrature rule ───────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = 5, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# Run gradient-informed algorithms
# ══════════════════════════════════════════════════════════════════════════════

print("=== SVGD ===")
trajectories_svgd = svgd(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== GI-ALDI ===")
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== EKS ===")
trajectories_eks = eks(
    model, n_particles=n_particles, n_steps=n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds, gradient_decay=gradient_decay,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-QG ===")
msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    partial(mc_quad_rule, N_quad=10),
    gradient_decay,
)
trajectories_msip_qg, traj_wts_msip_qg = msip(
    msip_quadgrad, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=(-1000, 1000),
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-GS-QG ===")
msip_quadgrad_gs = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    partial(mc_quad_rule, N_quad=10),
    gradient_decay,
)
trajectories_msip_gs_qg, traj_wts_msip_gs_qg = msip_gs(
    msip_quadgrad_gs, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=(-1000, 1000),
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-GMM ===")
msip_gmm = MSIPGMMGaussianKernel(
    weights=gmm_weights, means=gmm_means,
    covariances=gmm_covs, bandwidth=kernel_length_scale,
)
trajectories_msip_gmm, traj_wts_msip_gmm = msip(
    msip_gmm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-GS-GMM ===")
msip_gmm_gs = MSIPGMMGaussianKernel(
    weights=gmm_weights, means=gmm_means,
    covariances=gmm_covs, bandwidth=kernel_length_scale,
)
trajectories_msip_gs_gmm, traj_wts_msip_gs_gmm = msip_gs(
    msip_gmm_gs, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════════════════

algo_trajs = {
    "SVGD":          trajectories_svgd,
    "GI-ALDI":       trajectories_galdi,
    "EKS":           trajectories_eks,
    "MSIP-Fredholm": trajectories_msip_f,
    "MSIP-QG":       trajectories_msip_qg,
    "MSIP-GS-QG":    trajectories_msip_gs_qg,
    "MSIP-GMM":      trajectories_msip_gmm,
    "MSIP-GS-GMM":   trajectories_msip_gs_gmm,
}

algo_few_trajs = {
    "SVGD":          trajectories_svgd,
    "GI-ALDI":       trajectories_galdi,
    "EKS":           trajectories_eks,
    "MSIP-QG":       trajectories_msip_qg,
    "MSIP-GS-QG":    trajectories_msip_gs_qg,
}

algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
    "MSIP-QG":       traj_wts_msip_qg[-1],
    "MSIP-GS-QG":    traj_wts_msip_gs_qg[-1],
    "MSIP-GMM":      traj_wts_msip_gmm[-1],
    "MSIP-GS-GMM":   traj_wts_msip_gs_gmm[-1],
}

metrics        = {name: {"mmd": [], "ksd_rbf": [], "ksd_imq": []} for name in algo_trajs}
mmd_every      = 10
steps_recorded = list(range(0, n_steps, mmd_every))


# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════

def gmm_rbf_expectations(particles, weights, means, covs, bandwidth):
    sigma_sq = bandwidth ** 2
    K, D = means.shape
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)

    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for ell in range(K):
            cov_jl  = covs[j] + covs[ell] + 2 * sigma_sq * eye
            diff_jl = means[j] - means[ell]
            log_k   = -0.5 * (
                diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                + torch.logdet(2 * torch.pi * cov_jl)
            )
            Epp = Epp + weights[j] * weights[ell] * log_k.exp()

    smoothed_covs = covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - means[k].unsqueeze(0)
        log_k   = -0.5 * (
            torch.einsum("ni,ij,nj->n", diff_nk, torch.linalg.inv(smoothed_covs[k]), diff_nk)
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        )
        Exp = Exp + weights[k] * log_k.exp().mean()

    return Epp, Exp


def compute_mmd(particles, bandwidth):
    Epp, Exp = gmm_rbf_expectations(
        particles, gmm_weights, gmm_means, gmm_covs, bandwidth
    )
    Kxx = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bandwidth ** 2)
    ).mean()
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * kernel_length_scale ** 2))


def imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.1)


def compute_ksd(particles, log_p_grad_fn, kernel_fn):
    """Vectorized KSD U-statistic over all n*(n-1) pairs."""
    n = particles.shape[0]
    grads, _ = log_p_grad_fn(particles)

    xi_flat = particles.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    xj_flat = particles.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    si_flat = grads.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    sj_flat = grads.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)

    k_vals    = torch.vmap(kernel_fn)(xi_flat, xj_flat)
    grad_xi_k = torch.vmap(torch.func.grad(kernel_fn, argnums=0))(xi_flat, xj_flat)
    grad_xj_k = torch.vmap(torch.func.grad(kernel_fn, argnums=1))(xi_flat, xj_flat)

    def mixed_trace(a, b):
        return torch.func.jacfwd(
            torch.func.grad(kernel_fn, argnums=1), argnums=0
        )(a, b).diagonal().sum()

    trace_mixed = torch.vmap(mixed_trace)(xi_flat, xj_flat)

    h_flat = (
        (si_flat * sj_flat).sum(-1) * k_vals
        + (sj_flat * grad_xi_k).sum(-1)
        + (si_flat * grad_xj_k).sum(-1)
        + trace_mixed
    )

    mask = ~torch.eye(n, dtype=torch.bool, device=particles.device)
    return h_flat.reshape(n, n)[mask].sum().div(n * (n - 1)).clamp(min=0.0).sqrt().item()


# ══════════════════════════════════════════════════════════════════════════════
# MMD loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing MMD …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["mmd"].append(compute_mmd(traj[step_idx], kernel_length_scale))
print("MMD done.")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (RBF) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_rbf"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, rbf_fn)
        )
print("KSD (RBF) done.")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (IMQ) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_imq"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, imq_fn)
        )
print("KSD (IMQ) done.")


# ══════════════════════════════════════════════════════════════════════════════
# Contour grid
# ══════════════════════════════════════════════════════════════════════════════

Ngrid    = 100
xgrid    = torch.linspace(-10, 15, Ngrid)
ygrid    = torch.linspace(-10, 15, Ngrid)
X, Y     = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z        = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)


z_min = Z.max() - 20  # show 20 log-units below the peak
levels = torch.linspace(z_min, Z.max(), 20).cpu().numpy()
#ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=levels, alpha=0.6)

# ══════════════════════════════════════════════════════════════════════════════
# Individual particle PDFs (one per algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def plot_particles(ax, pts, wts=None, title=""):
    #ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20, alpha=0.6)
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=levels, alpha=0.6)
    c  = torch.abs(wts.cpu()) if wts is not None else None
    sc = ax.scatter(
        pts[:, 0].cpu(), pts[:, 1].cpu(),
        c=c, cmap="viridis", alpha=0.7, s=60,
        edgecolors="k", linewidths=0.4,
    )
    if wts is not None:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 15)
    ax.set_aspect(1.0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")


for name, traj in algo_trajs.items():
    pts = traj[-1]
    wts = algo_last_wts.get(name, None)
    if wts is not None:
        wts = wts / wts.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_particles(ax, pts, wts, title=f"{name} – step {n_steps}")
    plt.tight_layout()
    fname = f"gi_particles_{name.replace(' ', '_').replace('/', '-')}_sigma_"+str(kernel_length_scale)+".pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MMD vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["mmd"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("MMD (RBF kernel)")
ax.set_title("MMD vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_MMD_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_MMD_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_KSD_RBF_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_KSD_RBF_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_KSD_IMQ_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_KSD_IMQ_all_algorithms_sigma_"+str(kernel_length_scale)+".pdf")




# ══════════════════════════════════════════════════════════════════════════════
# MMD vs iteration - few algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["mmd"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("MMD (RBF kernel)")
ax.set_title("MMD vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_MMD_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_MMD_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) vs iteration - few algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_KSD_RBF_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_KSD_RBF_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")

# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) vs iteration - few algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – gradient-informed algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("gi_KSD_IMQ_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")
plt.close()
print("Saved gi_KSD_IMQ_notall_algorithms_sigma_"+str(kernel_length_scale)+".pdf")

print("All done.")