from functools import partial

import torch
import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import Tensor

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, gradfree_aldi, cbs, msip, msip_gs, kfrflow
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
    MSIPGMMGaussianKernel,
)
from nak_torch.tools.kernel import sqexp_kernel_elem as kernel_elem, sqexp_kernel_matrix

# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(1023921)


# ── Target: 3-component GMM ───────────────────────────────────────────────────
gmm_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
gmm_means = torch.stack([
    torch.tensor([6.2, -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([7.0,   0.0]),
])
gmm_covs = torch.stack([torch.eye(2) / 5.0] * 3)   # precision = 5 ⟹ cov = I/5
gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_batch         = torch.vmap(post_log_dens)
post_log_dens_grad_val      = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)


# ── Gaussian model wrapper needed by EKS / GF-ALDI ───────────────────────────
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

like_log_dens = lambda pt: -0.5 * (
    model.likelihood_precision
    * torch.linalg.norm(pt @ obs_op - model.true_obs, dim=-1) ** 2
).squeeze()


# ── Shared hyper-parameters ───────────────────────────────────────────────────
n_steps     = 500
n_particles = 25
lr          = 0.5
lr_msip     = 80e-3

kernel_length_scale = 2.8
kernel_diag_infl    = 1e-8
gradient_decay      = 1.0
bounds              = (-100.0, 100.0)

init_particles = torch.randn((n_particles, 2)) / \
    model.prior_precision + torch.tensor([3.2, -5.0])


# ── Quadrature rule ───────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = 50, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# Run all algorithms
# ══════════════════════════════════════════════════════════════════════════════

print("=== EKS ===")
trajectories_eks = eks(
    model, n_particles=n_particles, n_steps=n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== KFR-Flow ===")
trajectories_kfr = kfrflow(
    like_log_dens, n_particles, n_steps, 2,
    init_particles=init_particles,
    kernel_length_scale=1e-2,
    kernel_diag_infl=1e-5,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== GI-ALDI ===")
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== GF-ALDI ===")
trajectories_gfaldi = gradfree_aldi(
    model, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== CBS ===")
trajectories_cbs = cbs(
    post_log_dens, n_particles, n_steps, inverse_temp=0.95, dim=2,
    lr=lr, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

# ── MSIP – Fredholm ──────────────────────────────────────────────────────────
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

# ── MSIP – QuadGradientInformed ───────────────────────────────────────────────
print("=== MSIP-QuadGradientInformed ===")
msip_quadgrad = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    partial(mc_quad_rule, N_quad=50),
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

# ── MSIP – QuadGradientFree ───────────────────────────────────────────────────
print("=== MSIP-QuadGradientFree ===")
msip_quadgf = MSIPQuadGradientFree(
    post_log_dens_batch,
    partial(mc_quad_rule, N_quad=50),
)
trajectories_msip_qgf, traj_wts_msip_qgf = msip(
    msip_quadgf, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=(-1000.0, 1000.0),
    keep_all=True, compile_step=False, verbose=True,
)

# ── MSIP-GS – QuadGradientInformed ───────────────────────────────────────────
print("=== MSIP-GS (QuadGradientInformed) ===")
msip_quadgrad_gs = MSIPQuadGradientInformed(
    post_log_dens_grad_val_batch,
    partial(mc_quad_rule, N_quad=50),
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

# ── MSIP – GMM (closed-form) ──────────────────────────────────────────────────
print("=== MSIP-GMM ===")
msip_gmm = MSIPGMMGaussianKernel(
    weights=gmm_weights,
    means=gmm_means,
    covariances=gmm_covs,
    bandwidth=kernel_length_scale,
)
trajectories_msip_gmm, traj_wts_msip_gmm = msip(
    msip_gmm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: MMD and KSD
# ══════════════════════════════════════════════════════════════════════════════

def gmm_rbf_expectations(particles, weights, means, covs, bandwidth):
    """Analytical E[k(X,X')] and E[k(X,z)] for a GMM target with RBF kernel."""
    sigma_sq = bandwidth ** 2
    K, D = means.shape
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)

    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for ell in range(K):
            cov_jl = covs[j] + covs[ell] + 2 * sigma_sq * eye
            diff_jl = means[j] - means[ell]
            log_k = -0.5 * (
                diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                + torch.logdet(2 * torch.pi * cov_jl)
            )
            Epp = Epp + weights[j] * weights[ell] * log_k.exp()

    smoothed_covs = covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - means[k].unsqueeze(0)
        log_k = -0.5 * (
            torch.einsum(
                "ni,ij,nj->n", diff_nk,
                torch.linalg.inv(smoothed_covs[k]), diff_nk,
            )
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        )
        Exp = Exp + weights[k] * log_k.exp().mean()

    return Epp, Exp


def compute_mmd_rbf(particles, bandwidth):
    Epp, Exp = gmm_rbf_expectations(
        particles, gmm_weights, gmm_means, gmm_covs, bandwidth
    )
    Kxx = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bandwidth ** 2)
    ).mean()
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def rbf_kernel(x, y, h):
    return torch.exp(-torch.sum((x - y) ** 2) / (2 * h ** 2))


def imq_kernel(x, y, c=1.0, beta=-0.5):
    return (c + torch.sum((x - y) ** 2)) ** beta


def ksd_u_statistic(particles, log_p_grad_fn, kernel_fn):
    """
    KSD U-statistic with a given kernel.
    log_p_grad_fn : particles -> (grad_log_p, log_p)  (batched)
    """
    n = particles.shape[0]
    grads, _ = log_p_grad_fn(particles)          # (n, d)

    total = torch.tensor(0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, xj = particles[i], particles[j]
            si, sj = grads[i], grads[j]

            k_val = kernel_fn(xi, xj)

            # grad_xi k(xi, xj)
            grad_xi_k = torch.func.grad(lambda a: kernel_fn(a, xj))(xi)
            # grad_xj k(xi, xj)
            grad_xj_k = torch.func.grad(lambda b: kernel_fn(xi, b))(xj)
            # grad_xi grad_xj k(xi, xj)  (trace of mixed Hessian)
            def mixed(a):
                return torch.func.grad(lambda b: kernel_fn(a, b))(xj)
            trace_mixed = torch.func.jacfwd(mixed)(xi).diagonal().sum()

            h_val = (
                si @ sj * k_val
                + sj @ grad_xi_k
                + si @ grad_xj_k
                + trace_mixed
            )
            total = total + h_val

    return (total / (n * (n - 1))).clamp(min=0.0).sqrt().item()


# ══════════════════════════════════════════════════════════════════════════════
# Collect trajectories for metric computation
# ══════════════════════════════════════════════════════════════════════════════

# Map algorithm name -> trajectory list (each element is particles tensor)
algo_trajs = {
    "EKS":             trajectories_eks,
    "KFR-Flow":        trajectories_kfr,
    "GI-ALDI":         trajectories_galdi,
    "GF-ALDI":         trajectories_gfaldi,
    "CBS":             trajectories_cbs,
    "MSIP-Fredholm":   trajectories_msip_f,
    "MSIP-QG":         trajectories_msip_qg,
    "MSIP-QGF":        trajectories_msip_qgf,
    "MSIP-GS-QG":      trajectories_msip_gs_qg,
    "MSIP-GMM":        trajectories_msip_gmm,
}

# For algorithms that return weights alongside trajectories, store last weights
algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
    "MSIP-QG":       traj_wts_msip_qg[-1],
    "MSIP-QGF":      traj_wts_msip_qgf[-1],
    "MSIP-GS-QG":    traj_wts_msip_gs_qg[-1],
    "MSIP-GMM":      traj_wts_msip_gmm[-1],
}

mmd_every = max(1, n_steps // 100)
steps_recorded = list(range(0, n_steps, mmd_every))

print("Computing MMD and KSD for all algorithms …")

metrics = {name: {"mmd": [], "ksd_rbf": [], "ksd_imq": []} for name in algo_trajs}

rbf_fn  = lambda x, y: rbf_kernel(x, y, h=kernel_length_scale)
imq_fn  = lambda x, y: imq_kernel(x, y, c=1.0, beta=-0.5)

for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        pts = traj[step_idx]
        # MMD
        mmd_val = compute_mmd_rbf(pts, kernel_length_scale)
        metrics[name]["mmd"].append(mmd_val)
        # KSD – RBF
        ksd_r = ksd_u_statistic(pts, post_log_dens_grad_val_batch, rbf_fn)
        metrics[name]["ksd_rbf"].append(ksd_r)
        # KSD – IMQ
        ksd_i = ksd_u_statistic(pts, post_log_dens_grad_val_batch, imq_fn)
        metrics[name]["ksd_imq"].append(ksd_i)

print("Done computing metrics.")


# ══════════════════════════════════════════════════════════════════════════════
# Contour grid
# ══════════════════════════════════════════════════════════════════════════════

Ngrid = 100
xgrid = torch.linspace(-10, 15, Ngrid)
ygrid = torch.linspace(-10, 15, Ngrid)
X, Y  = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)


# ══════════════════════════════════════════════════════════════════════════════
# Individual particle plots (one PDF per algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def plot_particles(ax, pts, wts=None, title=""):
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20, alpha=0.6)
    c = wts.cpu() if wts is not None else None
    sc = ax.scatter(pts[:, 0].cpu(), pts[:, 1].cpu(),
                    c=c, cmap="viridis", alpha=0.7, s=60, edgecolors="k", linewidths=0.4)
    if wts is not None:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 15)
    ax.set_aspect(1.0)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")


for name, traj in algo_trajs.items():
    pts  = traj[-1]
    wts  = algo_last_wts.get(name, None)
    if wts is not None:
        wts = wts / wts.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_particles(ax, pts, wts, title=f"{name} – particles at step {n_steps}")
    plt.tight_layout()
    fname = f"particles_{name.replace(' ', '_').replace('/', '-')}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MMD vs iteration – all algorithms on one plot
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["mmd"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("MMD to target GMM (RBF kernel)")
ax.set_title("MMD vs Iteration – all algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("MMD_all_algorithms.pdf")
plt.close()
print("Saved MMD_all_algorithms.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) vs iteration – all algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – all algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("KSD_RBF_all_algorithms.pdf")
plt.close()
print("Saved KSD_RBF_all_algorithms.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) vs iteration – all algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – all algorithms")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("KSD_IMQ_all_algorithms.pdf")
plt.close()
print("Saved KSD_IMQ_all_algorithms.pdf")

print("All done.")
