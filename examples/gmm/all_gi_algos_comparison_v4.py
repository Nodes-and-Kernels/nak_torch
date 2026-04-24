"""
experiment_vs_M.py
──────────────────
Sweeps over M (number of particles) ∈ M_VALUES.
For each M, runs R independent initialisations of each algorithm.
Every algorithm is run to convergence (N_STEPS).
Metrics are evaluated at the final step only.

Algorithms
──────────
  SVGD | GI-ALDI | EKS | MSIP-QG (unweighted) | MSIP-QG (weighted)

Metrics  (all scalar)
─────────────────────
  MMD
  KSD-RBF, KSD-IMQ
  |Ê[f] − E_true[f]|  for f ∈ {x₁, x₂, x₁², x₂², x₁x₂}

Output
──────
  One median+IQR figure per metric  →  vsM_{metric}_median_iqr_sigma_*.pdf
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
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

M_VALUES  = [5, 10, 20, 50]
R_RUNS    = 10
N_STEPS   = 500

LR_SVGD   = 0.5
LR_ALDI   = 0.005 / 3
LR_EKS    = 0.005 / 3
LR_MSIP   = 0.5

KERNEL_LS      = 0.2
KERNEL_DIAG    = 1e-8
GRADIENT_DECAY = 1.0
BOUNDS_MSIP    = (-1000.0, 1000.0)
N_QUAD         = 10

INIT_MEAN = torch.tensor([8.0, 8.0])
INIT_STD  = 1.0
BASE_SEED = 314159
SIGMA_TAG = str(KERNEL_LS)

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
obs_op = torch.randn(2, 5)
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


# ══════════════════════════════════════════════════════════════════════════════
# TRUE GMM INTEGRALS  (closed form)
# ══════════════════════════════════════════════════════════════════════════════
# E[x_i]     = sum_k w_k * mu_k[i]
# E[x_i^2]   = sum_k w_k * (Sigma_k[i,i] + mu_k[i]^2)
# E[x_1 x_2] = sum_k w_k * (Sigma_k[0,1] + mu_k[0]*mu_k[1])

TRUE_INTEGRALS = {
    "x1":    sum(gmm_weights[k].item() * gmm_means[k][0].item()
                 for k in range(3)),
    "x2":    sum(gmm_weights[k].item() * gmm_means[k][1].item()
                 for k in range(3)),
    "x1_sq": sum(gmm_weights[k].item() * (gmm_covs[k][0, 0] + gmm_means[k][0] ** 2).item()
                 for k in range(3)),
    "x2_sq": sum(gmm_weights[k].item() * (gmm_covs[k][1, 1] + gmm_means[k][1] ** 2).item()
                 for k in range(3)),
    "x1x2":  sum(gmm_weights[k].item() * (gmm_covs[k][0, 1] + gmm_means[k][0] * gmm_means[k][1]).item()
                 for k in range(3)),
}

# Scalar test functions  f: (N, 2) → (N,)
TEST_FUNCTIONS = {
    "x1":    lambda pts: pts[:, 0],
    "x2":    lambda pts: pts[:, 1],
    "x1_sq": lambda pts: pts[:, 0] ** 2,
    "x2_sq": lambda pts: pts[:, 1] ** 2,
    "x1x2":  lambda pts: pts[:, 0] * pts[:, 1],
}


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── MMD (closed-form GMM cross-term) ────────────────────────────────────────
def _gmm_rbf_cross(particles, bw):
    sigma_sq = bw ** 2
    eye      = torch.eye(2, device=particles.device, dtype=particles.dtype)
    Epp      = torch.zeros((), device=particles.device, dtype=particles.dtype)
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
    Exp      = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for k in range(len(gmm_weights)):
        d     = particles - gmm_means[k]
        log_k = -0.5 * (
            torch.einsum("ni,ij,nj->n", d, torch.linalg.inv(smoothed[k]), d)
            + torch.logdet(2 * torch.pi * smoothed[k])
        )
        Exp = Exp + gmm_weights[k] * log_k.exp().mean()
    return Epp, Exp


def compute_mmd(particles, bw=KERNEL_LS):
    Epp, Exp = _gmm_rbf_cross(particles, bw)
    Kxx = torch.exp(
        -torch.cdist(particles, particles).pow(2) / (2 * bw ** 2)
    ).mean()
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def compute_mmd_weighted(particles, weights, bw=KERNEL_LS):
    w    = (weights / weights.sum()).clamp(min=0.0)
    Kxx  = (
        torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2))
        * w[:, None] * w[None, :]
    ).sum()
    Epp, _ = _gmm_rbf_cross(particles, bw)
    sigma_sq = bw ** 2
    eye      = torch.eye(2, device=particles.device, dtype=particles.dtype)
    smoothed = gmm_covs + sigma_sq * eye.unsqueeze(0)
    Exp_w    = torch.zeros((), device=particles.device, dtype=particles.dtype)
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
    n        = particles.shape[0]
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


# ── Scalar integral error ────────────────────────────────────────────────────
def integral_error(fname, particles, weights=None):
    vals = TEST_FUNCTIONS[fname](particles)
    if weights is not None:
        w   = weights / weights.sum()
        est = (w * vals).sum().item()
    else:
        est = vals.mean().item()
    return abs(est - TRUE_INTEGRALS[fname])


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

ALGO_NAMES = ["SVGD", "GI-ALDI", "EKS", "MSIP-QG (uw)", "MSIP-QG (w)"]


def run_one(M: int, seed: int) -> dict:
    torch.manual_seed(seed)
    init_p = INIT_MEAN + INIT_STD * torch.randn((M, 2))
    results = {}

    traj = svgd(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_SVGD, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["SVGD"] = (traj[-1], None)

    traj = grad_aldi(
        post_log_dens, M, N_STEPS, dim=2,
        lr=LR_ALDI, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["GI-ALDI"] = (traj[-1], None)

    traj = eks(
        ek_model, n_particles=M, n_steps=N_STEPS, dim=2,
        lr=LR_EKS, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["EKS"] = (traj[-1], None)

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
    results["MSIP-QG (uw)"] = (traj[-1], None)
    results["MSIP-QG (w)"]  = (traj[-1], wts[-1])

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════

raw_results = {}
for M in M_VALUES:
    print(f"\n{'='*60}\n  M = {M} particles\n{'='*60}")
    raw_results[M] = []
    for r in range(R_RUNS):
        seed = BASE_SEED + M * 10_000 + r * 997
        print(f"  run {r+1}/{R_RUNS}  (seed={seed})", flush=True)
        raw_results[M].append(run_one(M, seed))

print("\nSimulations done. Computing metrics …")

# ── Collect metrics ──────────────────────────────────────────────────────────

INTEGRAL_KEYS   = list(TEST_FUNCTIONS.keys())
DIVERGENCE_KEYS = ["mmd", "ksd_rbf", "ksd_imq"]
ALL_METRIC_KEYS = DIVERGENCE_KEYS + INTEGRAL_KEYS

metric_data = {
    mk: {algo: np.full((len(M_VALUES), R_RUNS), np.nan) for algo in ALGO_NAMES}
    for mk in ALL_METRIC_KEYS
}

for mi, M in enumerate(M_VALUES):
    for r in range(R_RUNS):
        run = raw_results[M][r]
        for algo in ALGO_NAMES:
            pts, wts = run[algo]
            is_w     = wts is not None

            metric_data["mmd"][algo][mi, r] = (
                compute_mmd_weighted(pts, wts) if is_w else compute_mmd(pts)
            )
            metric_data["ksd_rbf"][algo][mi, r] = compute_ksd(pts, _rbf_fn)
            metric_data["ksd_imq"][algo][mi, r] = compute_ksd(pts, _imq_fn)
            for fk in INTEGRAL_KEYS:
                metric_data[fk][algo][mi, r] = integral_error(
                    fk, pts, wts if is_w else None
                )

        print(f"  M={M}, run {r+1}/{R_RUNS} metrics done.", flush=True)

print("All metrics computed. Generating figures …")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING  —  one median+IQR figure per metric
# ══════════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLORS = {a: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, a in enumerate(ALGO_NAMES)}
ALGO_LS     = {"SVGD": "-", "GI-ALDI": "-", "EKS": "-",
               "MSIP-QG (uw)": "--", "MSIP-QG (w)": "-"}
ALGO_MARKER = {"SVGD": "o", "GI-ALDI": "s", "EKS": "^",
               "MSIP-QG (uw)": "D", "MSIP-QG (w)": "D"}

M_arr = np.array(M_VALUES, dtype=float)

METRIC_META = {
    "mmd"    : ("MMD (RBF kernel)",                       "MMD"),
    "ksd_rbf": ("KSD (RBF kernel)",                       "KSD_RBF"),
    "ksd_imq": ("KSD (IMQ kernel)",                       "KSD_IMQ"),
    "x1"     : (r"$|\hat{E}[x_1] - E[x_1]|$",            "integral_x1"),
    "x2"     : (r"$|\hat{E}[x_2] - E[x_2]|$",            "integral_x2"),
    "x1_sq"  : (r"$|\hat{E}[x_1^2] - E[x_1^2]|$",        "integral_x1sq"),
    "x2_sq"  : (r"$|\hat{E}[x_2^2] - E[x_2^2]|$",        "integral_x2sq"),
    "x1x2"   : (r"$|\hat{E}[x_1 x_2] - E[x_1 x_2]|$",   "integral_x1x2"),
}

for mk, (ylabel, filetag) in METRIC_META.items():
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in ALGO_NAMES:
        arr  = metric_data[mk][algo]          # (n_M, R_RUNS)
        med  = np.median(arr, axis=1)
        lo   = np.percentile(arr, 25, axis=1)
        hi   = np.percentile(arr, 75, axis=1)
        ax.loglog(M_arr, med,
                  color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                  linewidth=2.0, marker=ALGO_MARKER[algo], markersize=6,
                  label=algo)
        ax.fill_between(M_arr, lo, hi, alpha=0.20, color=ALGO_COLORS[algo])

    ax.set_xlabel("$M$  (number of particles)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"{ylabel}  vs  $M$\n(median ± IQR over {R_RUNS} runs)",
        fontsize=11,
    )
    ax.set_xticks(M_arr)
    ax.set_xticklabels([str(m) for m in M_VALUES])
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    fname = f"vsM_{filetag}_median_iqr_sigma_{SIGMA_TAG}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")

print("\nAll done.")