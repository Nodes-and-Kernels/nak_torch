"""
experiment_vs_M.py
──────────────────
Sweeps over M (number of particles) ∈ M_VALUES.
For each M, runs R independent initialisations of each algorithm.
Every algorithm is run to convergence (n_steps).
Metrics are evaluated **at the final step only**.

Algorithms compared
───────────────────
  SVGD | GI-ALDI | EKS | MSIP-QG (unweighted) | MSIP-QG (weighted)

Output figures (one pair per metric, x-axis = M on log scale)
──────────────────────────────────────────────────────────────
  *_median_iqr.pdf   – bold median line + IQR shaded band
  *_all_runs.pdf     – all individual thin lines + bold median
"""

from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

import nak_torch
from nak_torch.algorithms import grad_aldi, eks, msip, svgd
from nak_torch.algorithms.msip import MSIPQuadGradientInformed

# ── Device / dtype ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)

# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONFIGURATION  ← edit here
# ══════════════════════════════════════════════════════════════════════════════

M_VALUES   = [5, 10, 20, 50]   # particle counts to sweep
R_RUNS     = 10                # independent initialisations per M
N_STEPS    = 500                # steps to run each algorithm

LR_SVGD    = 0.01
LR_ALDI    = 0.005 / 3
LR_EKS     = 0.005 / 3
LR_MSIP    = 0.01

KERNEL_LS       = 1.0           # RBF / IMQ kernel length-scale
KERNEL_DIAG     = 1e-8          # MSIP diagonal inflation
GRADIENT_DECAY  = 1.0
BOUNDS_MSIP     = (-1000.0, 1000.0)
N_QUAD          = 10            # quadrature points for MSIP-QG

INIT_MEAN = torch.tensor([18.0, 18.0])
INIT_STD  = 1.0

BASE_SEED = 314159              # reproducibility

SIGMA_TAG = str(KERNEL_LS)     # used in output filenames

# ══════════════════════════════════════════════════════════════════════════════
# TARGET: 3-COMPONENT GMM
# ══════════════════════════════════════════════════════════════════════════════

gmm_weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])
gmm_means   = torch.stack([
    torch.tensor([6.2, -6.0]),
    torch.tensor([-4.0,  5.0]),
    torch.tensor([7.0,   0.0]),
])
gmm_covs = torch.stack([
    0.2 * torch.tensor([[1.0,  0.8], [0.8,  1.5]]),
    0.2 * torch.tensor([[2.0, -0.6], [-0.6, 0.5]]),
    0.2 * torch.tensor([[0.7,  0.4], [0.4,  1.2]]),
])
gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp   = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_grad_val       = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# ── EKS Gaussian model ──────────────────────────────────────────────────────
torch.manual_seed(BASE_SEED)
obs_op       = torch.randn(2, 5)
obs_op.div_(obs_op.norm(dim=1, keepdim=True))
forward_model = lambda particles: particles @ obs_op
true_obs      = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) + 20.0

ek_model = nak_torch.GaussianModel(
    forward_model,
    likelihood_precision=10.0,
    prior_precision=0.9,
    true_obs=true_obs,
    is_vectorized=True,
)

# ── Quadrature rule ─────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts

# ── True GMM moments ────────────────────────────────────────────────────────
TRUE_MEAN = (gmm_weights[:, None] * gmm_means).sum(0)

def _true_cov():
    mu  = TRUE_MEAN
    cov = torch.zeros(2, 2)
    for k in range(len(gmm_weights)):
        d   = gmm_means[k] - mu
        cov = cov + gmm_weights[k] * (gmm_covs[k] + d.outer(d))
    return cov

TRUE_COV = _true_cov()

# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── MMD (closed-form GMM cross-term) ───────────────────────────────────────
def _gmm_rbf_expectations(particles, bw):
    sigma_sq = bw ** 2
    D   = particles.shape[1]
    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(len(gmm_weights)):
        for l in range(len(gmm_weights)):
            cov_jl  = gmm_covs[j] + gmm_covs[l] + 2 * sigma_sq * eye
            diff_jl = gmm_means[j] - gmm_means[l]
            log_k   = -0.5 * (
                diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                + torch.logdet(2 * torch.pi * cov_jl)
            )
            Epp = Epp + gmm_weights[j] * gmm_weights[l] * log_k.exp()
    smoothed = gmm_covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(len(gmm_weights)):
        d     = particles - gmm_means[k]
        log_k = -0.5 * (
            torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(smoothed[k]), d)
            + torch.logdet(2 * torch.pi * smoothed[k])
        )
        Exp = Exp + gmm_weights[k] * log_k.exp().mean()
    return Epp, Exp


def compute_mmd(particles, bw=KERNEL_LS):
    Epp, Exp = _gmm_rbf_expectations(particles, bw)
    Kxx = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bw ** 2)
    ).mean()
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def compute_mmd_weighted(particles, weights, bw=KERNEL_LS):
    """MMD where the particle measure is weighted."""
    w    = (weights / weights.sum()).clamp(min=0.0)
    Kxx  = (
        torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2))
        * w[:, None] * w[None, :]
    ).sum()
    Epp, _ = _gmm_rbf_expectations(particles, bw)   # p-p term unchanged
    sigma_sq = bw ** 2
    eye      = torch.eye(particles.shape[1], device=particles.device, dtype=particles.dtype)
    smoothed = gmm_covs + sigma_sq * eye.unsqueeze(0)
    Exp_w = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(len(gmm_weights)):
        d     = particles - gmm_means[k]
        log_k = -0.5 * (
            torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(smoothed[k]), d)
            + torch.logdet(2 * torch.pi * smoothed[k])
        )
        Exp_w = Exp_w + gmm_weights[k] * (w * log_k.exp()).sum()
    return (Kxx + Epp - 2 * Exp_w).clamp(min=0.0).sqrt().item()


# ── KSD ─────────────────────────────────────────────────────────────────────
def _rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * KERNEL_LS ** 2))

def _imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.5)


def compute_ksd(particles, kernel_fn):
    n     = particles.shape[0]
    grads, _ = post_log_dens_grad_val_batch(particles)

    xi = particles.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    xj = particles.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    si = grads.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    sj = grads.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)

    k_vals    = torch.vmap(kernel_fn)(xi, xj)
    grad_xi_k = torch.vmap(torch.func.grad(kernel_fn, argnums=0))(xi, xj)
    grad_xj_k = torch.vmap(torch.func.grad(kernel_fn, argnums=1))(xi, xj)

    def mixed_trace(a, b):
        return torch.func.jacfwd(
            torch.func.grad(kernel_fn, argnums=1), argnums=0
        )(a, b).diagonal().sum()

    trace_mixed = torch.vmap(mixed_trace)(xi, xj)

    h = (
        (si * sj).sum(-1) * k_vals
        + (sj * grad_xi_k).sum(-1)
        + (si * grad_xj_k).sum(-1)
        + trace_mixed
    )
    mask = ~torch.eye(n, dtype=torch.bool, device=particles.device)
    return h.reshape(n, n)[mask].sum().div(n * (n - 1)).clamp(min=0.0).sqrt().item()


# ── Moment errors ────────────────────────────────────────────────────────────
def moment1_error(particles, weights=None):
    if weights is not None:
        w = weights / weights.sum()
        m = (w[:, None] * particles).sum(0)
    else:
        m = particles.mean(0)
    return (m - TRUE_MEAN).norm().item()


def moment2_error(particles, weights=None):
    """Frobenius error of the second-moment matrix E[XX^T] (not covariance)."""
    if weights is not None:
        w  = weights / weights.sum()
        M2 = (w[:, None, None] * particles[:, :, None] * particles[:, None, :]).sum(0)
    else:
        M2 = (particles[:, :, None] * particles[:, None, :]).mean(0)
    TRUE_M2 = TRUE_COV + TRUE_MEAN.outer(TRUE_MEAN)
    return (M2 - TRUE_M2).norm().item()


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

ALGO_NAMES = ["SVGD", "GI-ALDI", "EKS", "MSIP-QG (uw)", "MSIP-QG (w)"]


def run_one(M: int, seed: int) -> dict:
    """
    Run all algorithms with M particles, seeded by `seed`.
    Returns a dict: algo_name → (final_particles, final_weights_or_None)
    """
    torch.manual_seed(seed)
    init_p = INIT_MEAN + INIT_STD * torch.randn((M, 2))

    results = {}

    # ── SVGD ──────────────────────────────────────────────────────────────
    traj = svgd(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_SVGD, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["SVGD"] = (traj[-1], None)

    # ── GI-ALDI ───────────────────────────────────────────────────────────
    traj = grad_aldi(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_ALDI, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["GI-ALDI"] = (traj[-1], None)

    # ── EKS ───────────────────────────────────────────────────────────────
    traj = eks(
        ek_model, n_particles=M, n_steps=N_STEPS, dim=2,
        lr=LR_EKS, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["EKS"] = (traj[-1], None)

    # ── MSIP-QG ───────────────────────────────────────────────────────────
    msip_qg = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=N_QUAD),
        GRADIENT_DECAY,
    )
    traj, wts = msip(
        msip_qg, M, N_STEPS, dim=2,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    final_pts = traj[-1]
    final_wts = wts[-1]
    results["MSIP-QG (uw)"] = (final_pts, None)
    results["MSIP-QG (w)"]  = (final_pts, final_wts)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════

# raw_results[M][r][algo] = (pts, wts_or_None)
raw_results = {}

for M in M_VALUES:
    print(f"\n{'='*60}")
    print(f"  M = {M} particles")
    print(f"{'='*60}")
    raw_results[M] = []
    for r in range(R_RUNS):
        seed = BASE_SEED + M * 10_000 + r * 997
        print(f"  run {r+1}/{R_RUNS}  (seed={seed})", flush=True)
        raw_results[M].append(run_one(M, seed))

print("\nAll simulations done. Computing metrics …")

# ── Collect metrics ──────────────────────────────────────────────────────────
# metric_data[metric][algo] → array of shape (len(M_VALUES), R_RUNS)

METRIC_KEYS = ["mmd", "ksd_rbf", "ksd_imq", "moment1", "moment2"]
metric_data = {
    mk: {algo: np.full((len(M_VALUES), R_RUNS), np.nan)
         for algo in ALGO_NAMES}
    for mk in METRIC_KEYS
}

for mi, M in enumerate(M_VALUES):
    for r in range(R_RUNS):
        run = raw_results[M][r]
        for algo in ALGO_NAMES:
            pts, wts = run[algo]
            is_w     = (wts is not None)

            metric_data["mmd"][algo][mi, r] = (
                compute_mmd_weighted(pts, wts) if is_w else compute_mmd(pts)
            )
            metric_data["ksd_rbf"][algo][mi, r] = compute_ksd(pts, _rbf_fn)
            metric_data["ksd_imq"][algo][mi, r] = compute_ksd(pts, _imq_fn)
            metric_data["moment1"][algo][mi, r]  = moment1_error(pts, wts if is_w else None)
            metric_data["moment2"][algo][mi, r]  = moment2_error(pts, wts if is_w else None)

        print(f"  M={M}, run {r+1}/{R_RUNS} metrics done.", flush=True)

print("All metrics computed. Generating figures …")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLORS = {algo: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, algo in enumerate(ALGO_NAMES)}
ALGO_LS     = {
    "SVGD":         "-",
    "GI-ALDI":      "-",
    "EKS":          "-",
    "MSIP-QG (uw)": "--",
    "MSIP-QG (w)":  "-",
}
ALGO_MARKER = {
    "SVGD":         "o",
    "GI-ALDI":      "s",
    "EKS":          "^",
    "MSIP-QG (uw)": "D",
    "MSIP-QG (w)":  "D",
}

M_arr = np.array(M_VALUES)

METRIC_META = {
    "mmd":     ("MMD (RBF kernel)",                    "MMD"),
    "ksd_rbf": ("KSD (RBF kernel)",                    "KSD_RBF"),
    "ksd_imq": ("KSD (IMQ kernel)",                    "KSD_IMQ"),
    "moment1": ("‖Ê[X] − E[X]‖₂  (mean error)",       "moment1_error"),
    "moment2": ("‖Ê[XX^T] − E[XX^T]‖_F  (2nd moment)","moment2_error"),
}


def _add_reference_lines(ax):
    """Overlay M^{-1/2} and M^{-1} reference lines."""
    x0, x1 = M_arr[0], M_arr[-1]
    xs = np.array([x0, x1], dtype=float)
    for exp, lbl in [(-0.5, r"$M^{-1/2}$"), (-1.0, r"$M^{-1}$")]:
        ys = xs ** exp
        # normalise to pass through median of last algo at first M
        ys = ys / ys[0]   # relative shape only; will be rescaled per axis
        ax.plot(xs, ys * 1e-1, color="gray", linestyle=":", linewidth=0.9,
                alpha=0.6, label=lbl)


for mk, (ylabel, filetag) in METRIC_META.items():

    # ── Figure 1: median + IQR band ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in ALGO_NAMES:
        arr    = metric_data[mk][algo]          # (n_M, R_RUNS)
        med    = np.median(arr, axis=1)
        lo     = np.percentile(arr, 25, axis=1)
        hi     = np.percentile(arr, 75, axis=1)
        color  = ALGO_COLORS[algo]
        ls     = ALGO_LS[algo]
        marker = ALGO_MARKER[algo]
        ax.loglog(M_arr, med, color=color, linestyle=ls, linewidth=2.0,
                  marker=marker, markersize=6, label=algo)
        ax.fill_between(M_arr, lo, hi, alpha=0.20, color=color)

    ax.set_xlabel("M  (number of particles)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel}\nvs number of particles  (median ± IQR, {R_RUNS} runs)",
                 fontsize=11)
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fname = f"vsM_{filetag}_median_iqr_sigma_{SIGMA_TAG}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

    # ── Figure 2: all individual runs (thin) + bold median ───────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in ALGO_NAMES:
        arr    = metric_data[mk][algo]
        med    = np.median(arr, axis=1)
        color  = ALGO_COLORS[algo]
        ls     = ALGO_LS[algo]
        marker = ALGO_MARKER[algo]
        # thin individual runs
        for r in range(R_RUNS):
            ax.loglog(M_arr, arr[:, r], color=color, linestyle=ls,
                      linewidth=0.5, alpha=0.35, marker=marker, markersize=3)
        # bold median on top
        ax.loglog(M_arr, med, color=color, linestyle=ls, linewidth=2.5,
                  marker=marker, markersize=7, label=algo)

    ax.set_xlabel("M  (number of particles)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel}\nvs number of particles  (all {R_RUNS} runs + bold median)",
                 fontsize=11)
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fname = f"vsM_{filetag}_all_runs_sigma_{SIGMA_TAG}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

print("\nAll done.")