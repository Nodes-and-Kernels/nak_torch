"""
experiment_iteration_plots_highd.py
───────────────────────────────────
This is the high-dimensional analogue of your 2D plotting script.

It keeps the high-dimensional five-component axis-GMM target from
experiment_vs_M_msip_variants_highd.py, but produces the same kind of plots as
in your 2D script:

  1. MMD vs iteration
  2. KSD-RBF vs iteration
  3. KSD-IMQ vs iteration
  4. 2D projection particle plots on selected coordinate planes

The target is

    pi = (1/5) sum_{k=1}^5 N(alpha e_k, TARGET_COV_SCALE I_d),

so this file assumes d >= 5.

Algorithms
──────────
  SVGD | MSIP-Fredholm | MSIP-QG | MSIP-GS-QG | MSIP-GMM | MSIP-GS-GMM

Notes
─────
  • MMD is evaluated against the exact GMM target using the RBF kernel.
  • For MSIP variants, MMD uses the final/current MSIP weights.
  • KSD is evaluated on the support points only, as in your previous script.
  • Particle plots are projections, e.g. (x1,x2), (x1,x3), etc.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from nak_torch.algorithms import msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPGMMGaussianKernel,
    MSIPQuadGradientInformed,
)
from nak_torch.tools.quadrature import spherical_struct_radial_Laguerre


# ── Device / dtype ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR = Path("figs_highd_iteration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Axis-GMM with five modes alpha e_1, ..., alpha e_5 requires d >= 5.
D_VALUES = [5]
M_VALUES = [50]

N_STEPS = 500
METRIC_EVERY = 10
BASE_SEED = 314159

LR_SVGD = 0.5
LR_MSIP = 0.5

SCALE_KERNEL_WITH_DIM = False
KERNEL_LS_BASE = 0.5
KERNEL_DIAG = 1e-6
GRADIENT_DECAY = 1.0
BOUNDS_MSIP = (-1000.0, 1000.0)
N_QUAD = 1

MODE_SEPARATION_ALPHA = 5.5
TARGET_COV_SCALE = 0.5
ANISOTROPY_FACTOR = 3.0

# Initialization: particles are initialized from the same five-axis geometry.
# INIT_ALPHA = MODE_SEPARATION_ALPHA means initialized around the target modes.
# INIT_ALPHA = 0 initializes near the origin but with random component labels.
INIT_ALPHA = 20.0
INIT_STD = 1.0







# D_VALUES = [10]          # try [5, 10, 20]
# M_values = [5, 10, 15, 20, 50, 100]

# T = 500
# R = 20

# # Learning rates.
# lr = 0.5
# lr_aldi = 0.005 / 3
# lr_msip = 0.5

# # Kernel bandwidth. If SCALE_KERNEL_WITH_DIM=True, use sigma_base * sqrt(d).
# SCALE_KERNEL_WITH_DIM = False
# kernel_length_scale_base = 0.5
# kernel_diag_infl = 1e-6
# gradient_decay = 1.0
# bounds = (-1000.0, 1000.0)

# # Quad rule used by MSIP-QG variants.
# N_QUAD = 1

# # Target component means are MODE_SEPARATION_ALPHA * e_i, i=1,...,5.
# N_COMPONENTS = 5
# MODE_SEPARATION_ALPHA = 5.5

# # Covariance parameters.
# TARGET_COV_SCALE = 0.5
# ANISOTROPY_FACTOR = 3.0

# # Initialization: particles around INIT_ALPHA * e_1.
# INIT_ALPHA = 20.0
# init_std = 1.0

# base_seed = 314159





N_COMPONENTS = 5

ALGO_NAMES = [
    "SVGD",
    "MSIP-Fredholm",
    #"MSIP-QG",
    #"MSIP-GS-QG",
    #"MSIP-GMM",
    #"MSIP-GS-GMM",
]

# Coordinate projections to plot. Each pair is zero-indexed internally.
PROJECTION_PAIRS = [(0, 1), (0, 2), (0, 4)]

# If True, the weight colors in particle plots use projected nonnegative weights.
# If False, they use abs(raw weights), matching your earlier 2D script more closely.
PROJECT_WEIGHTS_FOR_PLOTS = False


# These globals are set by setup_gmm(d).
gmm_weights = None
gmm_means = None
gmm_covs = None
gmm_precisions = None
gmm_logdets = None
LOG_2PI = None
KERNEL_LS = None
SIGMA_TAG = None

post_log_dens_grad_val = None
post_log_dens_grad_val_batch = None


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def normalize_nonnegative(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp(min=0.0)
    return w / w.sum().clamp_min(1e-300)


def project_simplex(w: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    if w.ndim != 1:
        raise ValueError("project_simplex expects a 1D tensor.")

    u, _ = torch.sort(w, descending=True)
    cssv = torch.cumsum(u, dim=0) - z
    idx = torch.arange(1, w.numel() + 1, device=w.device, dtype=w.dtype)
    cond = u - cssv / idx > 0
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    tau = cssv[rho] / (rho + 1).to(w.dtype)
    return torch.clamp(w - tau, min=0.0)


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def effective_kernel_ls(d: int) -> float:
    if SCALE_KERNEL_WITH_DIM:
        return float(KERNEL_LS_BASE * np.sqrt(d))
    return float(KERNEL_LS_BASE)


# ══════════════════════════════════════════════════════════════════════════════
# TARGET: HIGH-DIMENSIONAL FIVE-COMPONENT AXIS GMM
# ══════════════════════════════════════════════════════════════════════════════

def make_axis_gmm(d: int, alpha: float = MODE_SEPARATION_ALPHA):
    if d < N_COMPONENTS:
        raise ValueError(
            f"The five-component axis-GMM requires d >= {N_COMPONENTS}; got d={d}."
        )

    means = torch.zeros((N_COMPONENTS, d), dtype=torch.get_default_dtype())
    for k in range(N_COMPONENTS):
        means[k, k] = alpha

    eye = torch.eye(d, dtype=torch.get_default_dtype())
    covs = TARGET_COV_SCALE * eye.unsqueeze(0).repeat(N_COMPONENTS, 1, 1)
    return means, covs


def setup_gmm(d: int):
    global gmm_weights, gmm_means, gmm_covs, gmm_precisions, gmm_logdets, LOG_2PI
    global KERNEL_LS, SIGMA_TAG
    global post_log_dens_grad_val, post_log_dens_grad_val_batch

    gmm_weights = torch.ones(N_COMPONENTS, dtype=torch.get_default_dtype()) / N_COMPONENTS
    gmm_means, gmm_covs = make_axis_gmm(d, alpha=MODE_SEPARATION_ALPHA)
    gmm_precisions = torch.linalg.inv(gmm_covs)
    gmm_logdets = torch.linalg.slogdet(gmm_covs).logabsdet
    LOG_2PI = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.get_default_dtype()))

    KERNEL_LS = effective_kernel_ls(d)
    SIGMA_TAG = str(KERNEL_LS).replace(".", "p")

    post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
    post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)


def post_log_dens(pt: torch.Tensor):
    """
    Normalized log density of the current Gaussian mixture.
    Supports pt of shape (..., d).
    """
    d = gmm_means.shape[1]
    log_probs = []
    for mean, prec, logdet, w in zip(gmm_means, gmm_precisions, gmm_logdets, gmm_weights):
        diff = pt - mean
        quad = torch.einsum("...i,ij,...j->...", diff, prec, diff)
        lp = torch.log(w) - 0.5 * (quad + logdet + d * LOG_2PI)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# QUADRATURE RULE
# ══════════════════════════════════════════════════════════════════════════════

def spherical_quad(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    N_spherical = 2 * dim
    N_radial = max(1, int(N_quad / N_spherical))
    pts, wts = spherical_struct_radial_Laguerre(
        batch_size, N_spherical, dim, N_radial, dtype=torch.float64
    )
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

# def make_init_particles(M: int, d: int):
#     comp_ids = torch.randint(low=0, high=N_COMPONENTS, size=(M,))
#     means = torch.zeros((M, d), dtype=torch.get_default_dtype())
#     means[torch.arange(M), comp_ids] = INIT_ALPHA
#     return means + INIT_STD * torch.randn((M, d))

# def make_init_particles(M: int, d: int):
#     mu = INIT_ALPHA * torch.ones((d,), dtype=torch.get_default_dtype())
#     return mu + INIT_STD * torch.randn((M, d))


def make_init_particles(M: int, d: int):
    mu = torch.zeros((d,), dtype=torch.get_default_dtype())
    mu[0] = INIT_ALPHA
    return mu + INIT_STD * torch.randn((M, d))


# ══════════════════════════════════════════════════════════════════════════════
# MMD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_rbf_expectation(mu1, cov1, mu2, cov2, bw):
    """
    E[exp(-||X-Y||^2 / (2 bw^2))]
    for independent X ~ N(mu1,cov1), Y ~ N(mu2,cov2).
    """
    d = mu1.numel()
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=mu1.device, dtype=mu1.dtype)
    S = cov1 + cov2
    diff = mu1 - mu2

    det_term = torch.linalg.det(eye + S / sigma_sq).clamp_min(1e-300) ** (-0.5)
    quad_term = torch.exp(-0.5 * diff @ torch.linalg.solve(S + sigma_sq * eye, diff))
    return det_term * quad_term


def _gmm_rbf_cross(particles, bw):
    d = particles.shape[1]
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=particles.device, dtype=particles.dtype)

    Epp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for j in range(N_COMPONENTS):
        for ell in range(N_COMPONENTS):
            Epp = Epp + gmm_weights[j] * gmm_weights[ell] * gaussian_rbf_expectation(
                gmm_means[j], gmm_covs[j], gmm_means[ell], gmm_covs[ell], bw
            )

    Exp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for k in range(N_COMPONENTS):
        diff = particles - gmm_means[k]
        A = gmm_covs[k] + sigma_sq * eye
        det_term = torch.linalg.det(eye + gmm_covs[k] / sigma_sq).clamp_min(1e-300) ** (-0.5)
        quad = torch.einsum("ni,ij,nj->n", diff, torch.linalg.inv(A), diff)
        Exp = Exp + gmm_weights[k] * (det_term * torch.exp(-0.5 * quad)).mean()

    return Epp, Exp


def compute_mmd(particles, bw=None, unbiased=False):
    if bw is None:
        bw = KERNEL_LS

    K = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2))
    n = particles.shape[0]

    if unbiased and n > 1:
        Kxx = (K.sum() - K.diag().sum()) / (n * (n - 1))
    else:
        Kxx = K.mean()

    Epp, Exp = _gmm_rbf_cross(particles, bw)
    return (Kxx + Epp - 2 * Exp).clamp(min=0.0).sqrt().item()


def compute_mmd_weighted(particles, weights, bw=None):
    if bw is None:
        bw = KERNEL_LS

    w = normalize_nonnegative(weights)
    K = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bw ** 2))
    Kxx = (K * w[:, None] * w[None, :]).sum()

    d = particles.shape[1]
    sigma_sq = bw ** 2
    eye = torch.eye(d, device=particles.device, dtype=particles.dtype)

    Epp = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for j in range(N_COMPONENTS):
        for ell in range(N_COMPONENTS):
            Epp = Epp + gmm_weights[j] * gmm_weights[ell] * gaussian_rbf_expectation(
                gmm_means[j], gmm_covs[j], gmm_means[ell], gmm_covs[ell], bw
            )

    Exp_w = torch.zeros((), device=particles.device, dtype=particles.dtype)
    for k in range(N_COMPONENTS):
        diff = particles - gmm_means[k]
        A = gmm_covs[k] + sigma_sq * eye
        det_term = torch.linalg.det(eye + gmm_covs[k] / sigma_sq).clamp_min(1e-300) ** (-0.5)
        quad = torch.einsum("ni,ij,nj->n", diff, torch.linalg.inv(A), diff)
        kvals = det_term * torch.exp(-0.5 * quad)
        Exp_w = Exp_w + gmm_weights[k] * (w * kvals).sum()

    return (Kxx + Epp - 2 * Exp_w).clamp(min=0.0).sqrt().item()


# ══════════════════════════════════════════════════════════════════════════════
# KSD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _rbf_fn(x, y):
    return torch.exp(-((x - y) ** 2).sum() / (2 * KERNEL_LS ** 2))


def _imq_fn(x, y):
    return (1.0 + ((x - y) ** 2).sum()) ** (-0.5)


def compute_ksd(particles, kernel_fn):
    """
    U-statistic KSD over all n(n-1) off-diagonal pairs.
    For weighted MSIP variants, this evaluates only the support points.
    """
    n = particles.shape[0]
    if n <= 1:
        return np.nan

    grads, _ = post_log_dens_grad_val_batch(particles)

    xi = particles.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    xj = particles.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
    si = grads.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
    sj = grads.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)

    k_vals = torch.vmap(kernel_fn)(xi, xj)
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


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def run_algorithms(M: int, d: int, seed: int):
    torch.manual_seed(seed)
    init_p = make_init_particles(M, d)
    results = {}

    print("    === SVGD ===", flush=True)
    traj = svgd(
        post_log_dens, M, N_STEPS, dim=d,
        lr=LR_SVGD, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        keep_all=True, compile_step=False, verbose=False,
    )
    results["SVGD"] = (traj, None)

    # print("    === MSIP-QG ===", flush=True)
    # msip_qg = MSIPQuadGradientInformed(
    #     post_log_dens_grad_val_batch,
    #     partial(spherical_quad, N_quad=N_QUAD, dim=d),
    #     GRADIENT_DECAY,
    # )
    # traj, wts = msip(
    #     msip_qg, M, N_STEPS, dim=d,
    #     lr=LR_MSIP, init_particles=init_p,
    #     kernel_length_scale=KERNEL_LS,
    #     kernel_diag_infl=KERNEL_DIAG,
    #     bounds=BOUNDS_MSIP,
    #     keep_all=True, compile_step=False, verbose=False,
    # )
    # results["MSIP-QG"] = (traj, wts)

    # print("    === MSIP-GS-QG ===", flush=True)
    # msip_gs_qg = MSIPQuadGradientInformed(
    #     post_log_dens_grad_val_batch,
    #     partial(spherical_quad, N_quad=N_QUAD, dim=d),
    #     GRADIENT_DECAY,
    # )
    # traj, wts = msip_gs(
    #     msip_gs_qg, M, N_STEPS, dim=d,
    #     lr=LR_MSIP, init_particles=init_p,
    #     kernel_length_scale=KERNEL_LS,
    #     kernel_diag_infl=KERNEL_DIAG,
    #     bounds=BOUNDS_MSIP,
    #     keep_all=True, compile_step=False, verbose=False,
    # )
    # results["MSIP-GS-QG"] = (traj, wts)

    # print("    === MSIP-GMM ===", flush=True)
    # msip_gmm = MSIPGMMGaussianKernel(
    #     weights=gmm_weights,
    #     means=gmm_means,
    #     covariances=gmm_covs,
    #     bandwidth=KERNEL_LS,
    # )
    # traj, wts = msip(
    #     msip_gmm, M, N_STEPS, dim=d,
    #     lr=LR_MSIP, init_particles=init_p,
    #     kernel_length_scale=KERNEL_LS,
    #     kernel_diag_infl=KERNEL_DIAG,
    #     bounds=BOUNDS_MSIP,
    #     keep_all=True, compile_step=False, verbose=False,
    # )
    # results["MSIP-GMM"] = (traj, wts)

    # print("    === MSIP-GS-GMM ===", flush=True)
    # msip_gs_gmm = MSIPGMMGaussianKernel(
    #     weights=gmm_weights,
    #     means=gmm_means,
    #     covariances=gmm_covs,
    #     bandwidth=KERNEL_LS,
    # )
    # traj, wts = msip_gs(
    #     msip_gs_gmm, M, N_STEPS, dim=d,
    #     lr=LR_MSIP, init_particles=init_p,
    #     kernel_length_scale=KERNEL_LS,
    #     kernel_diag_infl=KERNEL_DIAG,
    #     bounds=BOUNDS_MSIP,
    #     keep_all=True, compile_step=False, verbose=False,
    # )
    # results["MSIP-GS-GMM"] = (traj, wts)

    print("    === MSIP-Fredholm ===", flush=True)
    msip_fredholm = MSIPFredholm(GRADIENT_DECAY, post_log_dens_grad_val_batch)
    traj, wts = msip(
        msip_fredholm, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=True, compile_step=False, verbose=False,
    )
    results["MSIP-Fredholm"] = (traj, wts)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# METRIC CURVES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metric_curves(results):
    steps = list(range(0, N_STEPS, METRIC_EVERY))
    curves = {a: {"mmd": [], "ksd_rbf": [], "ksd_imq": []} for a in ALGO_NAMES}

    print("    Computing metric curves...", flush=True)
    for step_idx in steps:
        print(f"      step {step_idx}", flush=True)
        for algo in ALGO_NAMES:
            traj, wts_traj = results[algo]
            pts = traj[step_idx]

            if wts_traj is None:
                curves[algo]["mmd"].append(compute_mmd(pts))
            else:
                curves[algo]["mmd"].append(compute_mmd_weighted(pts, wts_traj[step_idx]))

            curves[algo]["ksd_rbf"].append(compute_ksd(pts, _rbf_fn))
            curves[algo]["ksd_imq"].append(compute_ksd(pts, _imq_fn))

    return steps, curves


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ══════════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLORS = {a: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, a in enumerate(ALGO_NAMES)}
ALGO_LS = {
    "SVGD": "-",
    "MSIP-QG": "-",
    "MSIP-GS-QG": "--",
    "MSIP-Fredholm": "-",
    "MSIP-GMM": "-",
    "MSIP-GS-GMM": "--",
}
ALGO_MARKER = {
    "SVGD": "o",
    "MSIP-QG": "D",
    "MSIP-GS-QG": "D",
    "MSIP-Fredholm": "v",
    "MSIP-GMM": "^",
    "MSIP-GS-GMM": "^",
}


def plot_metric_curve(steps, curves, metric_key, ylabel, d: int, M: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGO_NAMES:
        vals = np.maximum(np.asarray(curves[algo][metric_key], dtype=float), 1e-16)
        ax.semilogy(
            steps, vals,
            color=ALGO_COLORS[algo],
            linestyle=ALGO_LS[algo],
            marker=ALGO_MARKER[algo],
            linewidth=2.0,
            markersize=4,
            markevery=max(1, len(steps) // 10),
            label=algo,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs iteration, d={d}, M={M}")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    fname = OUT_DIR / f"iter_d{d}_M{M}_{metric_key}_sigma_{SIGMA_TAG}.pdf"
    plt.savefig(fname)
    plt.close()
    print(f"    Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PARTICLE PROJECTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def projected_log_density_grid(i: int, j: int, grid_lim: tuple[float, float], n_grid: int):
    """
    Log density on the coordinate plane spanned by e_i and e_j,
    with all other coordinates fixed to zero.
    """
    lo, hi = grid_lim
    xgrid = torch.linspace(lo, hi, n_grid)
    ygrid = torch.linspace(lo, hi, n_grid)
    X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")

    d = gmm_means.shape[1]
    grid_pts = torch.zeros((n_grid * n_grid, d), dtype=torch.get_default_dtype())
    grid_pts[:, i] = X.flatten()
    grid_pts[:, j] = Y.flatten()
    Z = post_log_dens(grid_pts).reshape(n_grid, n_grid)
    return X, Y, Z


def plot_particles_projection(ax, pts, wts, i: int, j: int, X, Y, Z, title: str):
    z_min = Z.max() - 20
    levels = torch.linspace(z_min, Z.max(), 20).cpu().numpy()
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=levels, alpha=0.6)

    if wts is None:
        c = None
    elif PROJECT_WEIGHTS_FOR_PLOTS:
        c = project_simplex(wts).detach().cpu()
    else:
        c = torch.abs(wts.detach().cpu())

    sc = ax.scatter(
        pts[:, i].detach().cpu(),
        pts[:, j].detach().cpu(),
        c=c,
        cmap="viridis" if c is not None else None,
        alpha=0.75,
        s=55,
        edgecolors="k",
        linewidths=0.35,
    )
    if c is not None:
        plt.colorbar(sc, ax=ax, shrink=0.78, label="weight")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"x{i + 1}")
    ax.set_ylabel(f"x{j + 1}")
    ax.set_aspect(1.0)


def plot_final_particles(results, d: int, M: int):
    grid_hi = MODE_SEPARATION_ALPHA + 4.0 * np.sqrt(TARGET_COV_SCALE)
    grid_lim = (-grid_hi, grid_hi)
    n_grid = 120

    for i, j in PROJECTION_PAIRS:
        if i >= d or j >= d:
            continue

        X, Y, Z = projected_log_density_grid(i, j, grid_lim, n_grid)

        for algo in ALGO_NAMES:
            traj, wts_traj = results[algo]
            pts = traj[-1]
            wts = None if wts_traj is None else wts_traj[-1]

            fig, ax = plt.subplots(figsize=(5.3, 5.0))
            plot_particles_projection(
                ax, pts, wts, i, j, X, Y, Z,
                title=f"{algo}, d={d}, M={M}, projection (x{i + 1}, x{j + 1})",
            )
            plt.tight_layout()

            fname = OUT_DIR / (
                f"particles_d{d}_M{M}_{safe_name(algo)}_x{i + 1}x{j + 1}"
                f"_sigma_{SIGMA_TAG}.pdf"
            )
            plt.savefig(fname)
            plt.close()
            print(f"    Saved {fname}")


def plot_all_metrics(steps, curves, d: int, M: int):
    plot_metric_curve(steps, curves, "mmd", "MMD (RBF kernel)", d, M)
    plot_metric_curve(steps, curves, "ksd_rbf", "KSD (RBF kernel)", d, M)
    plot_metric_curve(steps, curves, "ksd_imq", "KSD (IMQ kernel)", d, M)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for d in D_VALUES:
        setup_gmm(d)
        print("\n" + "#" * 72)
        print(f"Dimension d={d}")
        print(f"alpha={MODE_SEPARATION_ALPHA}, init_alpha={INIT_ALPHA}")
        print(f"kernel length-scale sigma={KERNEL_LS}")
        print("#" * 72)

        for M in M_VALUES:
            seed = BASE_SEED + d * 1_000_000 + M * 10_000
            print("\n" + "=" * 72)
            print(f"Running d={d}, M={M}, seed={seed}")
            print("=" * 72)

            results = run_algorithms(M, d, seed)
            steps, curves = compute_metric_curves(results)

            plot_all_metrics(steps, curves, d, M)
            plot_final_particles(results, d, M)

    print("\nAll done.")
    print(f"Figures saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()