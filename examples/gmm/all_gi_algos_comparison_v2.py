from functools import partial

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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


# ── Target: 3-component GMM ───────────────────────────────────────────────────
gmm_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
gmm_means = torch.stack([
    torch.tensor([6.2, -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([7.0,   0.0]),
])
gmm_covs = torch.stack([
    0.2 * torch.tensor([[1.0,  0.8],
                        [0.8,  1.5]]),
    0.2 * torch.tensor([[2.0, -0.6],
                        [-0.6, 0.5]]),
    0.2 * torch.tensor([[0.7,  0.4],
                        [0.4,  1.2]])
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
n_particles = 10
lr          = 0.01
lr_msip     = 1e-2

kernel_length_scale = 1.0
kernel_diag_infl    = 1e-8
gradient_decay      = 1.0
bounds              = (-100.0, 100.0)

init_mean = torch.tensor([18.0, 18.0])
init_std  = 1.0

# Number of independent runs for the multi-run analysis
N_RUNS = 5


# ── Quadrature rule ───────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = 10, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


# ── True GMM moments ─────────────────────────────────────────────────────────
def true_gmm_mean():
    """E[X] = sum_k w_k * mu_k"""
    return (gmm_weights[:, None] * gmm_means).sum(0)


def true_gmm_cov():
    """E[X X^T] - E[X] E[X]^T via law of total variance"""
    mu = true_gmm_mean()
    cov = torch.zeros(2, 2)
    for k in range(len(gmm_weights)):
        diff = gmm_means[k] - mu
        cov = cov + gmm_weights[k] * (gmm_covs[k] + diff.outer(diff))
    return cov


TRUE_MEAN = true_gmm_mean()
TRUE_COV  = true_gmm_cov()


# ══════════════════════════════════════════════════════════════════════════════
# Helper: run one seed → return trajectories + weights
# ══════════════════════════════════════════════════════════════════════════════

def run_all_algorithms(init_particles):
    """Run all 'notall' algorithms + MSIP variants. Returns dict of trajectories
    and a separate dict of (traj, weights) for MSIP algorithms."""

    traj_svgd = svgd(
        post_log_dens, n_particles, n_steps, dim=2,
        lr=lr, init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        keep_all=True, compile_step=False, verbose=False,
    )

    traj_galdi = grad_aldi(
        post_log_dens, n_particles, n_steps, dim=2,
        lr=lr / 3, init_particles=init_particles,
        keep_all=True, compile_step=False, verbose=False,
    )

    traj_eks = eks(
        model, n_particles=n_particles, n_steps=n_steps, dim=2,
        lr=lr / 3, init_particles=init_particles,
        keep_all=True, compile_step=False, verbose=False,
    )

    msip_quadgrad = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=10),
        gradient_decay,
    )
    traj_msip_qg, wts_msip_qg = msip(
        msip_quadgrad, n_particles, n_steps, dim=2,
        lr=lr_msip, init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        kernel_diag_infl=kernel_diag_infl,
        bounds=(-1000, 1000),
        keep_all=True, compile_step=False, verbose=False,
    )

    msip_quadgrad_gs = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=10),
        gradient_decay,
    )
    traj_msip_gs_qg, wts_msip_gs_qg = msip_gs(
        msip_quadgrad_gs, n_particles, n_steps, dim=2,
        lr=lr_msip, init_particles=init_particles,
        kernel_length_scale=kernel_length_scale,
        kernel_diag_infl=kernel_diag_infl,
        bounds=(-1000, 1000),
        keep_all=True, compile_step=False, verbose=False,
    )

    trajs = {
        "SVGD":       traj_svgd,
        "GI-ALDI":    traj_galdi,
        "EKS":        traj_eks,
        "MSIP-QG":    traj_msip_qg,
        "MSIP-GS-QG": traj_msip_gs_qg,
    }
    msip_wts = {
        "MSIP-QG":    wts_msip_qg,
        "MSIP-GS-QG": wts_msip_gs_qg,
    }
    return trajs, msip_wts


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


# ── Moment error helpers ──────────────────────────────────────────────────────

def weighted_mean(particles, weights):
    """particles: (N, D), weights: (N,) — returns (D,)"""
    w = weights / weights.sum()
    return (w[:, None] * particles).sum(0)


def weighted_cov(particles, weights):
    """Weighted covariance (D, D)"""
    w = weights / weights.sum()
    mu = (w[:, None] * particles).sum(0)
    diff = particles - mu
    return (w[:, None] * diff).T @ diff


def unweighted_mean(particles):
    return particles.mean(0)


def unweighted_cov(particles):
    return torch.cov(particles.T)


def mean_error(particles, weights=None):
    if weights is not None:
        m = weighted_mean(particles, weights)
    else:
        m = unweighted_mean(particles)
    return (m - TRUE_MEAN).norm().item()


def cov_frob_error(particles, weights=None):
    if weights is not None:
        C = weighted_cov(particles, weights)
    else:
        C = unweighted_cov(particles)
    return (C - TRUE_COV).norm().item()


# ══════════════════════════════════════════════════════════════════════════════
# Contour grid
# ══════════════════════════════════════════════════════════════════════════════

Ngrid    = 100
xgrid    = torch.linspace(-10, 15, Ngrid)
ygrid    = torch.linspace(-10, 15, Ngrid)
X, Y     = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z        = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE RUN  (original behaviour — all 8 algorithms)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SINGLE RUN — all algorithms")
print("=" * 70)

init_particles_single = init_mean + init_std * torch.randn((n_particles, 2))

print("=== SVGD ===")
trajectories_svgd = svgd(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles_single,
    kernel_length_scale=kernel_length_scale,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== GI-ALDI ===")
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles_single,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== EKS ===")
trajectories_eks = eks(
    model, n_particles=n_particles, n_steps=n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles_single,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles_single,
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
    lr=lr_msip, init_particles=init_particles_single,
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
    lr=lr_msip, init_particles=init_particles_single,
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
    lr=lr_msip, init_particles=init_particles_single,
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
    lr=lr_msip, init_particles=init_particles_single,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True,
)


# ── Algorithm registries (single run) ────────────────────────────────────────

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
    "SVGD":       trajectories_svgd,
    "GI-ALDI":    trajectories_galdi,
    "EKS":        trajectories_eks,
    "MSIP-QG":    trajectories_msip_qg,
    "MSIP-GS-QG": trajectories_msip_gs_qg,
}

algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
    "MSIP-QG":       traj_wts_msip_qg[-1],
    "MSIP-GS-QG":    traj_wts_msip_gs_qg[-1],
    "MSIP-GMM":      traj_wts_msip_gmm[-1],
    "MSIP-GS-GMM":   traj_wts_msip_gs_gmm[-1],
}

mmd_every      = 10
steps_recorded = list(range(0, n_steps, mmd_every))
metrics        = {name: {"mmd": [], "ksd_rbf": [], "ksd_imq": []} for name in algo_trajs}


# ── Original metric loops (single run, all algorithms) ───────────────────────

print("\nComputing MMD …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["mmd"].append(compute_mmd(traj[step_idx], kernel_length_scale))
print("MMD done.")

print("Computing KSD (RBF) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_rbf"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, rbf_fn)
        )
print("KSD (RBF) done.")

print("Computing KSD (IMQ) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_imq"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, imq_fn)
        )
print("KSD (IMQ) done.")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN PARTICLE PLOTS  (all algorithms)
# ══════════════════════════════════════════════════════════════════════════════

def plot_particles(ax, pts, wts=None, title=""):
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=20, alpha=0.6)
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
    fname = f"gi_particles_{name.replace(' ', '_').replace('/', '-')}_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ── Original MMD / KSD figures (all algorithms) ───────────────────────────────

for metric_key, ylabel, filetag in [
    ("mmd",     "MMD (RBF kernel)",  "MMD"),
    ("ksd_rbf", "KSD (RBF kernel)",  "KSD_RBF"),
    ("ksd_imq", "KSD (IMQ kernel)",  "KSD_IMQ"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in algo_trajs:
        ax.semilogy(steps_recorded, metrics[name][metric_key], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Iteration – gradient-informed algorithms")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_all_algorithms_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

for metric_key, ylabel, filetag in [
    ("mmd",     "MMD (RBF kernel)",  "MMD"),
    ("ksd_rbf", "KSD (RBF kernel)",  "KSD_RBF"),
    ("ksd_imq", "KSD (IMQ kernel)",  "KSD_IMQ"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in algo_few_trajs:
        ax.semilogy(steps_recorded, metrics[name][metric_key], label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Iteration – gradient-informed algorithms")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_notall_algorithms_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# (i) MULTI-RUN ANALYSIS  — notall algorithms, N_RUNS independent init seeds
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"MULTI-RUN ANALYSIS  ({N_RUNS} runs, notall algorithms)")
print("=" * 70)

few_algo_names = ["SVGD", "GI-ALDI", "EKS", "MSIP-QG", "MSIP-GS-QG"]

# Storage: metric → algo → list-of-lists (one per run)
multi_metrics = {
    name: {"mmd": [], "ksd_rbf": [], "ksd_imq": []}
    for name in few_algo_names
}

for run_idx in range(N_RUNS):
    print(f"\n--- Run {run_idx + 1}/{N_RUNS} ---")
    run_seed = 100_000 + run_idx * 7919
    torch.manual_seed(run_seed)
    init_p = init_mean + init_std * torch.randn((n_particles, 2))
    run_trajs, _ = run_all_algorithms(init_p)

    for step_idx in steps_recorded:
        for name in few_algo_names:
            if step_idx == steps_recorded[0]:          # first step of this run
                multi_metrics[name]["mmd"].append([])
                multi_metrics[name]["ksd_rbf"].append([])
                multi_metrics[name]["ksd_imq"].append([])

    for step_idx in steps_recorded:
        slot = steps_recorded.index(step_idx)
        for name in few_algo_names:
            traj = run_trajs[name]
            multi_metrics[name]["mmd"][run_idx].append(
                compute_mmd(traj[step_idx], kernel_length_scale))
            multi_metrics[name]["ksd_rbf"][run_idx].append(
                compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, rbf_fn))
            multi_metrics[name]["ksd_imq"][run_idx].append(
                compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, imq_fn))

print("Multi-run metrics done.")

# ── Plot multi-run: median + IQR band  ───────────────────────────────────────

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for metric_key, ylabel, filetag in [
    ("mmd",     "MMD (RBF kernel)",  "MMD"),
    ("ksd_rbf", "KSD (RBF kernel)",  "KSD_RBF"),
    ("ksd_imq", "KSD (IMQ kernel)",  "KSD_IMQ"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, name in enumerate(few_algo_names):
        color = COLOR_CYCLE[ci % len(COLOR_CYCLE)]
        arr = np.array(multi_metrics[name][metric_key])   # (N_RUNS, n_steps_recorded)
        med  = np.median(arr, axis=0)
        lo   = np.percentile(arr, 25, axis=0)
        hi   = np.percentile(arr, 75, axis=0)
        ax.semilogy(steps_recorded, med, color=color, label=name, linewidth=1.8)
        ax.fill_between(steps_recorded, lo, hi, alpha=0.25, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel} vs Iteration – {N_RUNS} runs (median ± IQR)\n"
        f"notall algorithms"
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_notall_multirun_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# (ii) MSIP WEIGHTED vs UNWEIGHTED  — separate figure per metric
# ══════════════════════════════════════════════════════════════════════════════
# We use the single-run trajectories already computed above.

print("\n" + "=" * 70)
print("MSIP WEIGHTED vs UNWEIGHTED  (single run)")
print("=" * 70)

msip_algo_names = ["MSIP-QG", "MSIP-GS-QG"]
msip_traj_map   = {
    "MSIP-QG":    (trajectories_msip_qg,    traj_wts_msip_qg),
    "MSIP-GS-QG": (trajectories_msip_gs_qg, traj_wts_msip_gs_qg),
}

msip_wv_metrics = {
    name: {
        "mmd_uw":     [],
        "mmd_w":      [],
        "ksd_rbf_uw": [],
        "ksd_rbf_w":  [],
        "ksd_imq_uw": [],
        "ksd_imq_w":  [],
    }
    for name in msip_algo_names
}

print("Computing MSIP weighted/unweighted metrics …")
for step_idx in steps_recorded:
    for name in msip_algo_names:
        traj, wts_list = msip_traj_map[name]
        pts = traj[step_idx]
        wts = wts_list[step_idx]

        # unweighted (uniform weights)
        msip_wv_metrics[name]["mmd_uw"].append(compute_mmd(pts, kernel_length_scale))
        msip_wv_metrics[name]["ksd_rbf_uw"].append(
            compute_ksd(pts, post_log_dens_grad_val_batch, rbf_fn))
        msip_wv_metrics[name]["ksd_imq_uw"].append(
            compute_ksd(pts, post_log_dens_grad_val_batch, imq_fn))

        # weighted — we encode weights into a "resampled" effective metric where
        # the kernel expectations use the empirical weighted measure.
        # For KSD we fall back to the standard unweighted KSD (weights enter only
        # through the sample positions in MMD).
        # For MMD with weighted measure we replace the particle term:
        wts_norm = (wts / wts.sum()).clamp(min=0.0)
        # Weighted Kxx
        Kxx_w = (
            torch.exp(-torch.cdist(pts, pts).pow(2) / (2 * kernel_length_scale ** 2))
            * wts_norm[:, None]
            * wts_norm[None, :]
        ).sum()
        Epp, Exp_uw = gmm_rbf_expectations(
            pts, gmm_weights, gmm_means, gmm_covs, kernel_length_scale
        )
        # Weighted Exp: replace uniform mean with weighted mean
        sigma_sq = kernel_length_scale ** 2
        eye = torch.eye(2, device=pts.device, dtype=pts.dtype)
        smoothed_covs = gmm_covs + sigma_sq * eye.unsqueeze(0)
        Exp_w = torch.tensor(0.0, device=pts.device, dtype=pts.dtype)
        for k in range(len(gmm_weights)):
            diff_nk = pts - gmm_means[k].unsqueeze(0)
            log_k   = -0.5 * (
                torch.einsum("ni,ij,nj->n", diff_nk,
                             torch.linalg.inv(smoothed_covs[k]), diff_nk)
                + torch.logdet(2 * torch.pi * smoothed_covs[k])
            )
            Exp_w = Exp_w + gmm_weights[k] * (wts_norm * log_k.exp()).sum()

        mmd_w = (Kxx_w + Epp - 2 * Exp_w).clamp(min=0.0).sqrt().item()
        msip_wv_metrics[name]["mmd_w"].append(mmd_w)

        # KSD does not have a natural closed-form for arbitrary weights;
        # we use the same unweighted KSD (particles already move under weighted rule).
        msip_wv_metrics[name]["ksd_rbf_w"].append(
            compute_ksd(pts, post_log_dens_grad_val_batch, rbf_fn))
        msip_wv_metrics[name]["ksd_imq_w"].append(
            compute_ksd(pts, post_log_dens_grad_val_batch, imq_fn))

print("MSIP w/uw metrics done.")

# ── Plot weighted vs unweighted per metric ────────────────────────────────────

linestyles = {"uw": "--", "w": "-"}
labels     = {"uw": "unweighted", "w": "weighted"}

for metric_key, ylabel, filetag, w_key, uw_key in [
    ("mmd",     "MMD (RBF kernel)",  "MMD",     "mmd_w",      "mmd_uw"),
    ("ksd_rbf", "KSD (RBF kernel)",  "KSD_RBF", "ksd_rbf_w",  "ksd_rbf_uw"),
    ("ksd_imq", "KSD (IMQ kernel)",  "KSD_IMQ", "ksd_imq_w",  "ksd_imq_uw"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, name in enumerate(msip_algo_names):
        color = COLOR_CYCLE[ci % len(COLOR_CYCLE)]
        for variant, ls in linestyles.items():
            key = w_key if variant == "w" else uw_key
            ax.semilogy(
                steps_recorded,
                msip_wv_metrics[name][key],
                color=color, linestyle=ls, linewidth=1.8,
                label=f"{name} ({labels[variant]})",
            )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} – MSIP weighted vs unweighted")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_MSIP_weighted_vs_unweighted_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# (iii) MOMENT ERRORS  — notall algorithms + MSIP weighted & unweighted
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("MOMENT ERRORS  (notall algorithms + MSIP w/uw)")
print("=" * 70)

moment_names_base = ["SVGD", "GI-ALDI", "EKS", "MSIP-QG", "MSIP-GS-QG"]

# Build name → (traj, wts_or_None) map
moment_traj_map = {
    "SVGD":              (trajectories_svgd,       None),
    "GI-ALDI":           (trajectories_galdi,      None),
    "EKS":               (trajectories_eks,        None),
    "MSIP-QG (uw)":      (trajectories_msip_qg,   None),
    "MSIP-QG (w)":       (trajectories_msip_qg,   traj_wts_msip_qg),
    "MSIP-GS-QG (uw)":   (trajectories_msip_gs_qg, None),
    "MSIP-GS-QG (w)":    (trajectories_msip_gs_qg, traj_wts_msip_gs_qg),
}

moment_errors = {
    name: {"mean": [], "cov": []}
    for name in moment_traj_map
}

print("Computing moment errors …")
for step_idx in steps_recorded:
    for name, (traj, wts_list) in moment_traj_map.items():
        pts = traj[step_idx]
        wts = wts_list[step_idx] if wts_list is not None else None
        moment_errors[name]["mean"].append(mean_error(pts, wts))
        moment_errors[name]["cov"].append(cov_frob_error(pts, wts))
print("Moment errors done.")

# ── Define linestyle/color per name ──────────────────────────────────────────

_base_colors = {
    "SVGD":    0,
    "GI-ALDI": 1,
    "EKS":     2,
    "MSIP-QG (uw)":    3,
    "MSIP-QG (w)":     3,
    "MSIP-GS-QG (uw)": 4,
    "MSIP-GS-QG (w)":  4,
}
_base_ls = {
    "SVGD":    "-",
    "GI-ALDI": "-",
    "EKS":     "-",
    "MSIP-QG (uw)":    "--",
    "MSIP-QG (w)":     "-",
    "MSIP-GS-QG (uw)": "--",
    "MSIP-GS-QG (w)":  "-",
}

for moment_key, ylabel, filetag in [
    ("mean", "‖E[X]_approx − E[X]_true‖₂",  "mean_error"),
    ("cov",  "‖Cov_approx − Cov_true‖_F",    "cov_error"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in moment_traj_map:
        color = COLOR_CYCLE[_base_colors[name] % len(COLOR_CYCLE)]
        ls    = _base_ls[name]
        ax.semilogy(
            steps_recorded,
            moment_errors[name][moment_key],
            color=color, linestyle=ls, linewidth=1.8,
            label=name,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Moment error ({moment_key}) vs Iteration – notall algorithms\n"
                 f"(MSIP: solid = weighted, dashed = unweighted)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_notall_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-RUN MOMENT ERRORS  (notall algorithms + MSIP w/uw, across N_RUNS runs)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"MULTI-RUN MOMENT ERRORS  ({N_RUNS} runs)")
print("=" * 70)

multi_moment_names = list(moment_traj_map.keys())
multi_moment_errors = {
    name: {"mean": [], "cov": []}
    for name in multi_moment_names
}

for run_idx in range(N_RUNS):
    print(f"\n--- Moment run {run_idx + 1}/{N_RUNS} ---")
    run_seed = 200_000 + run_idx * 6271
    torch.manual_seed(run_seed)
    init_p = init_mean + init_std * torch.randn((n_particles, 2))
    run_trajs, run_wts = run_all_algorithms(init_p)

    # Build per-run moment_traj_map
    run_moment_map = {
        "SVGD":              (run_trajs["SVGD"],       None),
        "GI-ALDI":           (run_trajs["GI-ALDI"],    None),
        "EKS":               (run_trajs["EKS"],        None),
        "MSIP-QG (uw)":      (run_trajs["MSIP-QG"],   None),
        "MSIP-QG (w)":       (run_trajs["MSIP-QG"],   run_wts["MSIP-QG"]),
        "MSIP-GS-QG (uw)":   (run_trajs["MSIP-GS-QG"], None),
        "MSIP-GS-QG (w)":    (run_trajs["MSIP-GS-QG"], run_wts["MSIP-GS-QG"]),
    }

    for name in multi_moment_names:
        multi_moment_errors[name]["mean"].append([])
        multi_moment_errors[name]["cov"].append([])

    for step_idx in steps_recorded:
        for name, (traj, wts_list) in run_moment_map.items():
            pts = traj[step_idx]
            wts = wts_list[step_idx] if wts_list is not None else None
            multi_moment_errors[name]["mean"][run_idx].append(mean_error(pts, wts))
            multi_moment_errors[name]["cov"][run_idx].append(cov_frob_error(pts, wts))

print("Multi-run moment errors done.")

for moment_key, ylabel, filetag in [
    ("mean", "‖E[X]_approx − E[X]_true‖₂",  "mean_error"),
    ("cov",  "‖Cov_approx − Cov_true‖_F",    "cov_error"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in multi_moment_names:
        color = COLOR_CYCLE[_base_colors[name] % len(COLOR_CYCLE)]
        ls    = _base_ls[name]
        arr   = np.array(multi_moment_errors[name][moment_key])   # (N_RUNS, T)
        med   = np.median(arr, axis=0)
        lo    = np.percentile(arr, 25, axis=0)
        hi    = np.percentile(arr, 75, axis=0)
        ax.semilogy(steps_recorded, med, color=color, linestyle=ls,
                    linewidth=1.8, label=name)
        ax.fill_between(steps_recorded, lo, hi, alpha=0.18, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Moment error ({moment_key}) vs Iteration – {N_RUNS} runs (median ± IQR)\n"
        f"(MSIP: solid = weighted, dashed = unweighted)"
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"gi_{filetag}_notall_multirun_sigma_{kernel_length_scale}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


print("\nAll done.")