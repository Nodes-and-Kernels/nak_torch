"""
gmm_highd_final_cross_entropy.py
────────────────────────────────
High-dimensional version of the clean GMM final-cross-entropy benchmark.

This keeps the same structure as the 2D file:
  - sweep over particle counts M
  - R independent runs
  - final cross entropy only
  - errorbar plot of mean ± std

Target:
  Five-component Gaussian mixture in R^d, d >= 5.
  Means are alpha e_1, ..., alpha e_5.
  Covariances are diagonal anisotropic: each component has a larger variance
  along its own active axis.

Metric:
  Final cross entropy
      - E_{mu_T}[log pi]
  Here pi is the normalized high-dimensional GMM density.
"""

from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

import nak_torch
from nak_torch.algorithms import grad_aldi, msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPGMMGaussianKernel,
)
from nak_torch.tools.metrics import CrossEntropy


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

D_VALUES = [10]          # try [5, 10, 20]
M_values = [5, 10, 15, 20, 50, 100]

T = 500
R = 20

# Learning rates.
lr = 0.5
lr_aldi = 0.005 / 3
lr_msip = 0.5

# Kernel bandwidth. If SCALE_KERNEL_WITH_DIM=True, use sigma_base * sqrt(d).
SCALE_KERNEL_WITH_DIM = False
kernel_length_scale_base = 0.5
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-1000.0, 1000.0)

# Quad rule used by MSIP-QG variants.
N_QUAD = 1

# Target component means are MODE_SEPARATION_ALPHA * e_i, i=1,...,5.
N_COMPONENTS = 5
MODE_SEPARATION_ALPHA = 5.5

# Covariance parameters.
TARGET_COV_SCALE = 0.5
ANISOTROPY_FACTOR = 3.0

# Initialization: particles around INIT_ALPHA * e_1.
INIT_ALPHA = 20.0
init_std = 1.0

base_seed = 314159

# Algorithms to compare.
algo_names = [
    "SVGD",
#    "GI-ALDI",
    "MSIP-Fredholm",
    # Uncomment these if you also want the high-dimensional MSIP variants.
    # "MSIP-QG",
    # "MSIP-GS-QG",
    # "MSIP-GMM",
    # "MSIP-GS-GMM",
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


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-DIMENSIONAL FIVE-COMPONENT AXIS GMM
# ══════════════════════════════════════════════════════════════════════════════

def effective_kernel_ls(d: int) -> float:
    if SCALE_KERNEL_WITH_DIM:
        return float(kernel_length_scale_base * np.sqrt(d))
    return float(kernel_length_scale_base)


def make_axis_gmm(d: int, alpha: float = MODE_SEPARATION_ALPHA):
    r"""
    Build a five-component GMM in R^d with means

        mu_k = alpha e_k,    k = 1,...,5.

    Requires d >= 5. Coordinates 6,...,d have zero mean for all components.
    Each covariance is diagonal anisotropic: component k has larger variance
    along coordinate k.
    """
    if d < N_COMPONENTS:
        raise ValueError(f"Axis-GMM construction requires d >= {N_COMPONENTS}; got d={d}.")

    means = torch.zeros((N_COMPONENTS, d), dtype=torch.get_default_dtype())
    for k in range(N_COMPONENTS):
        means[k, k] = alpha

    covs = TARGET_COV_SCALE * torch.eye(
        d, dtype=torch.get_default_dtype()
    ).unsqueeze(0).repeat(N_COMPONENTS, 1, 1)

    for k in range(N_COMPONENTS):
        covs[k, k, k] = ANISOTROPY_FACTOR * TARGET_COV_SCALE

    return means, covs


def setup_gmm(d: int):
    global gmm_weights, gmm_means, gmm_covs, gmm_precisions, gmm_logdets, LOG_2PI
    global KERNEL_LS, SIGMA_TAG

    gmm_weights = torch.ones(N_COMPONENTS, dtype=torch.get_default_dtype()) / N_COMPONENTS
    gmm_means, gmm_covs = make_axis_gmm(d, alpha=MODE_SEPARATION_ALPHA)
    gmm_precisions = torch.linalg.inv(gmm_covs)
    gmm_logdets = torch.linalg.slogdet(gmm_covs).logabsdet
    LOG_2PI = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.get_default_dtype()))

    KERNEL_LS = effective_kernel_ls(d)
    SIGMA_TAG = str(KERNEL_LS).replace(".", "p")


# ══════════════════════════════════════════════════════════════════════════════
# TARGET LOG DENSITY
# ══════════════════════════════════════════════════════════════════════════════

def post_log_dens(pt: torch.Tensor):
    """
    Normalized log-density of the current high-dimensional Gaussian mixture.
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
post_log_dens_batch = torch.vmap(post_log_dens)

cross_entropy_metric = CrossEntropy(
    post_log_dens_batch,
    is_log_dens_vectorized=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# QUADRATURE RULES FOR OPTIONAL MSIP-QG VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_init_particles(n_particles: int, run_idx: int, M: int, d: int) -> torch.Tensor:
    """
    Reproducible initialization around INIT_ALPHA * e_1.
    This matches the high-dimensional file's current initialization.
    """
    torch.manual_seed(base_seed + d * 1_000_000 + 1000 * M + run_idx)
    mu = torch.zeros((d,), dtype=torch.get_default_dtype())
    mu[0] = INIT_ALPHA
    return mu + init_std * torch.randn((n_particles, d))


def final_cross_entropy(pts: torch.Tensor, wts: torch.Tensor | None = None) -> float:
    """Compute final cross entropy; normalize weights when present."""
    if wts is not None:
        wts = wts.clamp(min=0.0)
        wts = wts / wts.sum().clamp_min(1e-300)
    return cross_entropy_metric(pts, wts=wts).item()


def run_one_algorithm(algo_name: str, n_particles: int, init_particles: torch.Tensor, d: int) -> float:
    """
    Run one algorithm for T iterations and return only the final cross entropy.
    """
    if algo_name == "SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=d,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1])

    if algo_name == "GI-ALDI":
        trajectories = grad_aldi(
            post_log_dens,
            n_particles,
            T,
            dim=d,
            lr=lr_aldi,
            init_particles=init_particles.clone(),
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1])

    if algo_name == "MSIP-Fredholm":
        msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
        trajectories, traj_wts = msip(
            msip_fredholm,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            gradient_decay=gradient_decay,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    if algo_name == "MSIP-QG":
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=N_QUAD, dim=d),
            gradient_decay,
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    if algo_name == "MSIP-GS-QG":
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=N_QUAD, dim=d),
            gradient_decay,
        )
        trajectories, traj_wts = msip_gs(
            msip_qg,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    if algo_name == "MSIP-GMM":
        msip_gmm = MSIPGMMGaussianKernel(
            weights=gmm_weights,
            means=gmm_means,
            covariances=gmm_covs,
            bandwidth=KERNEL_LS,
        )
        trajectories, traj_wts = msip(
            msip_gmm,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    if algo_name == "MSIP-GS-GMM":
        msip_gmm = MSIPGMMGaussianKernel(
            weights=gmm_weights,
            means=gmm_means,
            covariances=gmm_covs,
            bandwidth=KERNEL_LS,
        )
        trajectories, traj_wts = msip_gs(
            msip_gmm,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=KERNEL_LS,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    for d in D_VALUES:
        setup_gmm(d)

        results = {
            M: {algo_name: [] for algo_name in algo_names}
            for M in M_values
        }

        print("Running high-dimensional GMM final-cross-entropy benchmark")
        print(f"d = {d}")
        print(f"T = {T}, R = {R}, M_values = {M_values}")
        print(f"lr = {lr}, lr_aldi = {lr_aldi}, lr_msip = {lr_msip}")
        print(f"kernel_length_scale = {KERNEL_LS}")
        print(f"mode separation alpha = {MODE_SEPARATION_ALPHA}, init alpha = {INIT_ALPHA}")
        print()

        for M in M_values:
            print(f"=== d = {d}, M = {M} ===")
            for run_idx in range(R):
                init_particles = make_init_particles(M, run_idx, M, d)

                for algo_name in algo_names:
                    ce = run_one_algorithm(algo_name, M, init_particles, d)
                    results[M][algo_name].append(ce)

                print(f"  run {run_idx + 1:02d}/{R} done", flush=True)
            print()

        # Report table.
        print(f"\nFinal cross entropy over R runs, d = {d}")
        print("smaller is better")
        print()
        print(f"{'M':>5}  {'Algorithm':<16}  {'mean':>14}  {'std':>14}")
        print("-" * 55)

        summary_rows = []
        for M in M_values:
            for algo_name in algo_names:
                vals = torch.tensor(results[M][algo_name], dtype=torch.float64)
                mean = vals.mean().item()
                std = vals.std(unbiased=True).item() if R > 1 else 0.0
                summary_rows.append((d, M, algo_name, mean, std))
                print(f"{M:5d}  {algo_name:<16}  {mean:14.6f}  {std:14.6f}")

        # Save raw and summary results.
        out_name = (
            "highd_gmm_final_cross_entropy_"
            f"d_{d}_T_{T}_R_{R}_sigma_{SIGMA_TAG}.pt"
        )

        torch.save(
            {
                "d": d,
                "T": T,
                "R": R,
                "M_values": M_values,
                "algo_names": algo_names,
                "results": results,
                "summary_rows": summary_rows,
                "lr": lr,
                "lr_aldi": lr_aldi,
                "lr_msip": lr_msip,
                "kernel_length_scale": KERNEL_LS,
                "kernel_diag_infl": kernel_diag_infl,
                "gradient_decay": gradient_decay,
                "bounds": bounds,
                "init_alpha": INIT_ALPHA,
                "init_std": init_std,
                "base_seed": base_seed,
                "mode_separation_alpha": MODE_SEPARATION_ALPHA,
                "target_cov_scale": TARGET_COV_SCALE,
                "anisotropy_factor": ANISOTROPY_FACTOR,
                "gmm_weights": gmm_weights.detach().cpu(),
                "gmm_means": gmm_means.detach().cpu(),
                "gmm_covs": gmm_covs.detach().cpu(),
            },
            out_name,
        )
        print(f"\nSaved raw results to {out_name}")

        # Plot mean final cross entropy vs M.
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo_name in algo_names:
            means = []
            stds = []
            for M in M_values:
                vals = torch.tensor(results[M][algo_name], dtype=torch.float64)
                means.append(vals.mean().item())
                stds.append(vals.std(unbiased=True).item() if R > 1 else 0.0)

            ax.errorbar(M_values, means, yerr=stds, marker="o", capsize=4, label=algo_name)

        ax.set_xlabel("Number of particles M")
        ax.set_ylabel(r"Final cross entropy  $-\mathbb{E}_{\mu_T}[\log \pi]$")
        ax.set_title(f"High-dimensional GMM final cross entropy, d={d}, T={T}, R={R}")
        ax.legend(fontsize=8)
        plt.tight_layout()

        fig_name = (
            "highd_gmm_final_cross_entropy_vs_M_"
            f"d_{d}_T_{T}_R_{R}_sigma_{SIGMA_TAG}.pdf"
        )
        plt.savefig(fig_name)
        plt.close()
        print(f"Saved {fig_name}")

    print("All done.")


if __name__ == "__main__":
    main()