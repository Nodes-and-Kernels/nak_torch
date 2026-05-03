"""
experiment_vs_M_msip_variants_highd.py
──────────────────────────────────────
High-dimensional extension of experiment_vs_M_msip_variants.py.

Sweeps over dimensions d in D_VALUES and number of particles M in M_VALUES.
For each (d, M), runs R independent initializations of each algorithm.
Every algorithm is run to convergence (N_STEPS).
Metrics are evaluated at the final step only.

Algorithms
──────────
  SVGD | GI-ALDI | MSIP-Fredholm | MSIP-QG | MSIP-GS-QG | MSIP-GMM | MSIP-GS-GMM

Target
──────
  Five-component Gaussian mixture in R^d, for d in {3, 5, 10} by default.
  The first two coordinates retain the original 2D GMM structure.
  The remaining coordinates add component-dependent mean shifts and diagonal anisotropy.

Metrics
───────
  MMD
  KSD-RBF, KSD-IMQ
  |Ehat[f] - E_true[f]| for f in:
    x_j, x_j^2 for all j=1,...,d
    selected cross moments x1*x2, x1*xd, x2*xd

Output
──────
  One median+IQR figure per metric and per dimension:
      vsM_d{d}_{metric}_median_iqr_sigma_*.pdf
"""

from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from nak_torch.algorithms import grad_aldi, msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPQuadGradientInformed,
    MSIPGMMGaussianKernel,
    MSIPFredholm,
)
from nak_torch.tools.quadrature import (
    spherical_MC_radial_Laguerre,
    spherical_struct_radial_Laguerre,
)

# ── Device / dtype ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def project_simplex(w: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """
    Euclidean projection of w onto {u >= 0, sum u = z}.
    Works for a 1D tensor w.
    """
    if w.ndim != 1:
        raise ValueError("project_simplex expects a 1D tensor.")

    u, _ = torch.sort(w, descending=True)
    cssv = torch.cumsum(u, dim=0) - z

    idx = torch.arange(1, w.numel() + 1, device=w.device, dtype=w.dtype)
    cond = u - cssv / idx > 0

    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    tau = cssv[rho] / (rho + 1).to(w.dtype)

    return torch.clamp(w - tau, min=0.0)


def normalize_nonnegative(weights: torch.Tensor) -> torch.Tensor:
    w = weights.clamp(min=0.0)
    return w / w.sum().clamp_min(1e-300)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

D_VALUES = [3]
M_VALUES = [5, 10, 20, 50]
R_RUNS = 10
N_STEPS = 500

LR_SVGD = 0.5
LR_ALDI = 0.005 / 3
LR_MSIP = 0.5

# If SCALE_KERNEL_WITH_DIM=True, the effective bandwidth is KERNEL_LS_BASE * sqrt(d).
# If False, the effective bandwidth is KERNEL_LS_BASE for all d.
SCALE_KERNEL_WITH_DIM = False
KERNEL_LS_BASE = 0.5

KERNEL_DIAG = 1e-6
GRADIENT_DECAY = 1.0
BOUNDS_MSIP = (-1000.0, 1000.0)
N_QUAD = 1

INIT_STD = 1.0
INIT_LOCATION = 15.0
BASE_SEED = 314159

N_COMPONENTS = 5

ALGO_NAMES = [
    "SVGD",
    "GI-ALDI",
    "MSIP-Fredholm",
    "MSIP-QG",
    "MSIP-GS-QG",
    "MSIP-GMM",
    "MSIP-GS-GMM",
]

# These globals are set by setup_gmm(d).
gmm_weights = None

gmm_means = None
gmm_covs = None
gmm_precisions = None
gmm_logdets = None
LOG_2PI = None
KERNEL_LS = None
SIGMA_TAG = None
TRUE_INTEGRALS = None
TEST_FUNCTIONS = None


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-DIMENSIONAL FIVE-COMPONENT GMM
# ══════════════════════════════════════════════════════════════════════════════

BASE_MEANS_2D = 2.0 * torch.stack([
    torch.tensor([6.2, -6.0]),
    torch.tensor([-4.0, 5.0]),
    torch.tensor([7.0, 3.0]),
    torch.tensor([-6.5, -4.5]),
    torch.tensor([1.0, 7.0]),
])

BASE_COVS_2D = torch.stack([
    torch.tensor([[1.5, 0.1], [0.1, 0.5]]),
    torch.tensor([[2.0, -0.6], [-0.6, 0.5]]),
    torch.tensor([[0.7, 0.4], [0.4, 1.2]]),
    torch.tensor([[1.3, -0.5], [-0.5, 0.9]]),
    torch.tensor([[0.6, 0.35], [0.35, 1.6]]),
])


def effective_kernel_ls(d: int) -> float:
    if SCALE_KERNEL_WITH_DIM:
        return float(KERNEL_LS_BASE * np.sqrt(d))
    return float(KERNEL_LS_BASE)


def make_high_d_gmm(d: int):
    """
    Build a five-component GMM in R^d.

    The first two coordinates retain the original 2D GMM. Extra coordinates are
    component-dependent, so the high-dimensional mixture remains genuinely
    multimodal and not just a trivial product extension.
    """
    if d < 2:
        raise ValueError("This construction expects d >= 2.")

    means = torch.zeros((N_COMPONENTS, d), dtype=torch.get_default_dtype())
    covs = torch.zeros((N_COMPONENTS, d, d), dtype=torch.get_default_dtype())

    means[:, :2] = BASE_MEANS_2D
    covs[:, :2, :2] = BASE_COVS_2D

    if d > 2:
        extra = d - 2

        # Mode-dependent shifts in the extra coordinates.
        # Shape: (5, extra). Each component gets a distinct high-d signature.
        base_shifts = torch.linspace(-2.0, 2.0, N_COMPONENTS, dtype=torch.get_default_dtype())
        coord_profile = torch.linspace(0.7, 1.3, extra, dtype=torch.get_default_dtype())
        means[:, 2:] = base_shifts[:, None] * coord_profile[None, :]

        # Mild diagonal anisotropy in the additional coordinates.
        extra_vars = torch.linspace(0.6, 1.4, extra, dtype=torch.get_default_dtype())
        for k in range(N_COMPONENTS):
            covs[k, 2:, 2:] = torch.diag(extra_vars)

    return means, covs


def setup_gmm(d: int):
    global gmm_weights, gmm_means, gmm_covs, gmm_precisions, gmm_logdets, LOG_2PI
    global KERNEL_LS, SIGMA_TAG, TRUE_INTEGRALS, TEST_FUNCTIONS

    gmm_weights = torch.ones(N_COMPONENTS, dtype=torch.get_default_dtype()) / N_COMPONENTS
    gmm_means, gmm_covs = make_high_d_gmm(d)
    gmm_precisions = torch.linalg.inv(gmm_covs)
    gmm_logdets = torch.linalg.slogdet(gmm_covs).logabsdet
    LOG_2PI = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.get_default_dtype()))

    KERNEL_LS = effective_kernel_ls(d)
    SIGMA_TAG = str(KERNEL_LS).replace(".", "p")

    TRUE_INTEGRALS = {}
    TEST_FUNCTIONS = {}

    # Coordinate means and second moments.
    for j in range(d):
        TRUE_INTEGRALS[f"x{j + 1}"] = sum(
            gmm_weights[k].item() * gmm_means[k, j].item()
            for k in range(N_COMPONENTS)
        )
        TRUE_INTEGRALS[f"x{j + 1}_sq"] = sum(
            gmm_weights[k].item()
            * (gmm_covs[k, j, j] + gmm_means[k, j] ** 2).item()
            for k in range(N_COMPONENTS)
        )

        TEST_FUNCTIONS[f"x{j + 1}"] = lambda pts, j=j: pts[:, j]
        TEST_FUNCTIONS[f"x{j + 1}_sq"] = lambda pts, j=j: pts[:, j] ** 2

    # Selected cross moments. This avoids producing O(d^2) figures.
    candidate_pairs = [(0, 1), (0, d - 1), (1, d - 1)]
    seen_pairs = set()
    for i, j in candidate_pairs:
        if i >= j or (i, j) in seen_pairs:
            continue
        seen_pairs.add((i, j))
        key = f"x{i + 1}x{j + 1}"
        TRUE_INTEGRALS[key] = sum(
            gmm_weights[k].item()
            * (gmm_covs[k, i, j] + gmm_means[k, i] * gmm_means[k, j]).item()
            for k in range(N_COMPONENTS)
        )
        TEST_FUNCTIONS[key] = lambda pts, i=i, j=j: pts[:, i] * pts[:, j]


# ══════════════════════════════════════════════════════════════════════════════
# TARGET LOG DENSITY
# ══════════════════════════════════════════════════════════════════════════════

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


post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)


# ══════════════════════════════════════════════════════════════════════════════
# QUADRATURE RULES
# ══════════════════════════════════════════════════════════════════════════════

def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad_(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    dimension = dim
    N_spherical = 2 * dimension
    N_radial = max(1, int(N_quad / N_spherical))
    pts, wts = spherical_MC_radial_Laguerre(
        batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
    )
    return pts, wts


def spherical_quad(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    dimension = dim
    N_spherical = 2 * dimension
    N_radial = max(1, int(N_quad / N_spherical))
    pts, wts = spherical_struct_radial_Laguerre(
        batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
    )
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# MMD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_rbf_expectation(mu1, cov1, mu2, cov2, bw):
    """
    E[exp(-||X-Y||^2 / (2 bw^2))] for independent Gaussians
    X ~ N(mu1, cov1), Y ~ N(mu2, cov2).
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
    """
    Returns:
      Epp = E_{X,X'~pi}[k(X,X')]
      Exp = E_{X~emp, Y~pi}[k(X,Y)]
    for k(x,y)=exp(-||x-y||^2/(2 bw^2)).
    """
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
    # Alternative:
    # w = project_simplex(weights)

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
    Unweighted KSD. For weighted MSIP variants, this evaluates only the final
    support points, as in the original script.
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
# SCALAR INTEGRAL ERROR
# ══════════════════════════════════════════════════════════════════════════════

def integral_error(fname, particles, weights=None):
    vals = TEST_FUNCTIONS[fname](particles)
    if weights is not None:
        w = normalize_nonnegative(weights)
        # Alternative:
        # w = project_simplex(weights)
        est = (w * vals).sum().item()
    else:
        est = vals.mean().item()
    return abs(est - TRUE_INTEGRALS[fname])


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def make_init_particles(M: int, d: int):
    init_mean = torch.full((d,), INIT_LOCATION, dtype=torch.get_default_dtype())
    return init_mean + INIT_STD * torch.randn((M, d))


def run_one(M: int, seed: int, d: int) -> dict:
    torch.manual_seed(seed)
    init_p = make_init_particles(M, d)
    results = {}

    traj = svgd(
        post_log_dens, M, N_STEPS, dim=d,
        lr=LR_SVGD, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["SVGD"] = (traj[-1], None)

    traj = grad_aldi(
        post_log_dens, M, N_STEPS, dim=d,
        lr=LR_ALDI, init_particles=init_p,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["GI-ALDI"] = (traj[-1], None)

    msip_qg = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=N_QUAD, dim=d),
        GRADIENT_DECAY,
    )
    traj, wts = msip(
        msip_qg, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-QG"] = (traj[-1], wts[-1])

    msip_gs_qg = MSIPQuadGradientInformed(
        post_log_dens_grad_val_batch,
        partial(mc_quad_rule, N_quad=N_QUAD, dim=d),
        GRADIENT_DECAY,
    )
    traj, wts = msip_gs(
        msip_gs_qg, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-GS-QG"] = (traj[-1], wts[-1])

    msip_gmm = MSIPGMMGaussianKernel(
        weights=gmm_weights,
        means=gmm_means,
        covariances=gmm_covs,
        bandwidth=KERNEL_LS,
    )
    traj, wts = msip(
        msip_gmm, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-GMM"] = (traj[-1], wts[-1])

    msip_gs_gmm = MSIPGMMGaussianKernel(
        weights=gmm_weights,
        means=gmm_means,
        covariances=gmm_covs,
        bandwidth=KERNEL_LS,
    )
    traj, wts = msip_gs(
        msip_gs_gmm, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-GS-GMM"] = (traj[-1], wts[-1])

    msip_fredholm = MSIPFredholm(GRADIENT_DECAY, post_log_dens_grad_val_batch)
    traj, wts = msip(
        msip_fredholm, M, N_STEPS, dim=d,
        lr=LR_MSIP, init_particles=init_p,
        kernel_length_scale=KERNEL_LS,
        kernel_diag_infl=KERNEL_DIAG,
        bounds=BOUNDS_MSIP,
        keep_all=False, compile_step=False, verbose=False,
    )
    results["MSIP-Fredholm"] = (traj[-1], wts[-1])

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ALGO_COLORS = {a: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, a in enumerate(ALGO_NAMES)}
ALGO_LS = {
    "SVGD": "-",
    "GI-ALDI": "-",
    "MSIP-QG": "-",
    "MSIP-Fredholm": "-",
    "MSIP-GS-QG": "--",
    "MSIP-GMM": "-",
    "MSIP-GS-GMM": "--",
}
ALGO_MARKER = {
    "SVGD": "o",
    "GI-ALDI": "s",
    "MSIP-QG": "D",
    "MSIP-GS-QG": "D",
    "MSIP-Fredholm": "v",
    "MSIP-GMM": "^",
    "MSIP-GS-GMM": "^",
}

M_arr = np.array(M_VALUES, dtype=float)
PLOT_EPS = 1e-16


def pretty_metric_label(key: str) -> str:
    if key == "mmd":
        return "MMD (RBF kernel)"
    if key == "ksd_rbf":
        return "KSD (RBF kernel)"
    if key == "ksd_imq":
        return "KSD (IMQ kernel)"
    return rf"$|\hat{{E}}[{key}] - E[{key}]|$"


def build_metric_meta(integral_keys):
    metric_meta = {
        "mmd": ("MMD (RBF kernel)", "MMD"),
        "ksd_rbf": ("KSD (RBF kernel)", "KSD_RBF"),
        "ksd_imq": ("KSD (IMQ kernel)", "KSD_IMQ"),
    }
    for fk in integral_keys:
        metric_meta[fk] = (pretty_metric_label(fk), f"integral_{fk}")
    return metric_meta


def plot_metric(metric_data, metric_meta, d: int):
    for mk, (ylabel, filetag) in metric_meta.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo in ALGO_NAMES:
            arr = metric_data[mk][algo]
            med = np.maximum(np.nanmedian(arr, axis=1), PLOT_EPS)
            lo = np.maximum(np.nanpercentile(arr, 25, axis=1), PLOT_EPS)
            hi = np.maximum(np.nanpercentile(arr, 75, axis=1), PLOT_EPS)

            ax.loglog(
                M_arr, med,
                color=ALGO_COLORS[algo], linestyle=ALGO_LS[algo],
                linewidth=2.0, marker=ALGO_MARKER[algo], markersize=6,
                label=algo,
            )
            ax.fill_between(M_arr, lo, hi, alpha=0.20, color=ALGO_COLORS[algo])

        ax.set_xlabel("$M$  (number of particles)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{ylabel} vs $M$, $d={d}$\n(median ± IQR over {R_RUNS} runs)",
            fontsize=11,
        )
        ax.set_xticks(M_arr)
        ax.set_xticklabels([str(m) for m in M_VALUES])
        ax.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        fname = f"vsM_d{d}_{filetag}_median_iqr_sigma_{SIGMA_TAG}.pdf"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for d in D_VALUES:
        print(f"\n\n{'#' * 70}\nDIMENSION d = {d}\n{'#' * 70}")
        setup_gmm(d)
        print(f"Using kernel length-scale sigma = {KERNEL_LS}")

        raw_results = {}
        for M in M_VALUES:
            print(f"\n{'=' * 60}\n  d = {d}, M = {M} particles\n{'=' * 60}")
            raw_results[M] = []
            for r in range(R_RUNS):
                seed = BASE_SEED + d * 1_000_000 + M * 10_000 + r * 997
                print(f"  run {r + 1}/{R_RUNS}  (seed={seed})", flush=True)
                raw_results[M].append(run_one(M, seed, d))

        print(f"\nd={d}: simulations done. Computing metrics …")

        integral_keys = list(TEST_FUNCTIONS.keys())
        divergence_keys = ["mmd", "ksd_rbf", "ksd_imq"]
        all_metric_keys = divergence_keys + integral_keys

        metric_data = {
            mk: {algo: np.full((len(M_VALUES), R_RUNS), np.nan) for algo in ALGO_NAMES}
            for mk in all_metric_keys
        }

        for mi, M in enumerate(M_VALUES):
            for r in range(R_RUNS):
                run = raw_results[M][r]
                for algo in ALGO_NAMES:
                    pts, wts = run[algo]
                    is_w = wts is not None

                    metric_data["mmd"][algo][mi, r] = (
                        compute_mmd_weighted(pts, wts) if is_w else compute_mmd(pts)
                    )
                    metric_data["ksd_rbf"][algo][mi, r] = compute_ksd(pts, _rbf_fn)
                    metric_data["ksd_imq"][algo][mi, r] = compute_ksd(pts, _imq_fn)

                    for fk in integral_keys:
                        metric_data[fk][algo][mi, r] = integral_error(
                            fk, pts, wts if is_w else None
                        )

                print(f"  d={d}, M={M}, run {r + 1}/{R_RUNS} metrics done.", flush=True)

        print(f"d={d}: all metrics computed. Generating figures …")
        metric_meta = build_metric_meta(integral_keys)
        plot_metric(metric_data, metric_meta, d)

    print("\nAll done.")


if __name__ == "__main__":
    main()