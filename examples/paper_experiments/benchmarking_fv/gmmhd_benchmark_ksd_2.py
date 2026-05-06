from __future__ import annotations

from functools import partial
from pathlib import Path
import sys
import os
import csv
from typing import Any

import torch

# If this file is placed inside your experiments folder, this keeps the same
# relative import style as your original script.
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from nak_torch.algorithms import grad_aldi, msip, svgd
from nak_torch.algorithms.deepensembles import deepensembles
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)
from nak_torch.tools.metrics import CrossEntropy, KernelSteinDiscrepancy


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EXP_NAME = "gmmhd"
PARAM = ""
if PARAM != "":
    EXP_NAME = EXP_NAME + "_" + str(PARAM)

ROOT_DIR = Path("results") / EXP_NAME
METRICS_DIR = ROOT_DIR / "metrics"
SUMMARIES_DIR = ROOT_DIR / "summaries"
PLOTS_DIR = ROOT_DIR / "plots"
FINAL_STATES_DIR = ROOT_DIR / "final_states"
PARTICLES_DIR = FINAL_STATES_DIR / "particles"
WEIGHTS_DIR = FINAL_STATES_DIR / "weights"
COMBINED_STATES_DIR = FINAL_STATES_DIR / "combined_particles_weights"

for directory in [
    ROOT_DIR,
    METRICS_DIR,
    SUMMARIES_DIR,
    PLOTS_DIR,
    FINAL_STATES_DIR,
    PARTICLES_DIR,
    WEIGHTS_DIR,
    COMBINED_STATES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Dimensions and particles.
# The five-component axis-GMM requires d >= 5.
D_VALUES = [5, 10, 20]
M_values = [25]

# Number of iterations and independent runs.
T = 1000
R = 5

# Target GMM parameters.
N_COMPONENTS = 5
MODE_SEPARATION_ALPHA = 7.5
TARGET_COV_SCALE = 0.5
ANISOTROPY_FACTOR = 1.5

# Initialization: particles around INIT_ALPHA e_1.
INIT_ALPHA = 0.0
init_std = 1.0

# Learning rates.
lr = 0.1
lr_msip = 0.1
lr_aldi = 0.01

# Kernel and MSIP parameters used by the algorithms themselves.
# The evaluation metrics below use SIGMAS independently.
SCALE_KERNEL_WITH_DIM = False
kernel_length_scale_base = 0.1
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-1000.0, 1000.0)

# Evaluation metrics.
SIGMAS = [0.01, 0.05, 0.1]
KSD_KERNELS = ["RBF", "IMQ"]

# Quadrature parameters.
N_QUAD = 10
N_QUAD_GF = 10

base_seed = 314159

algo_names = [
    "a-SVGD",
    "SVGD",
    "GI-ALDI",
    "MSIP-Fredholm",
    "MSIP-GI",
    "MSIP-GF",
    "DeepEnsembles",
]


# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS SET BY setup_gmm(d)
# ══════════════════════════════════════════════════════════════════════════════

gmm_weights = None
gmm_means = None
gmm_covs = None
gmm_precisions = None
gmm_logdets = None
LOG_2PI = None

kernel_length_scale = None
post_log_dens_grad_val = None
post_log_dens_grad_val_batch = None
post_log_dens_batch = None
post_grad_log_dens_batch = None
cross_entropy_metric = None


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-DIMENSIONAL ANISOTROPIC AXIS GMM
# ══════════════════════════════════════════════════════════════════════════════

def effective_kernel_length_scale(d: int) -> float:
    if SCALE_KERNEL_WITH_DIM:
        return float(kernel_length_scale_base * (d ** 0.5))
    return float(kernel_length_scale_base)


def make_axis_anisotropic_gmm(d: int, alpha: float = MODE_SEPARATION_ALPHA):
    """
    Five-component GMM in dimension d.

    Means:
        m_k = alpha e_k, k = 1, ..., 5.

    Covariances:
        diagonal anisotropic matrices. For component k, the k-th coordinate
        has variance TARGET_COV_SCALE * ANISOTROPY_FACTOR, while the other
        coordinates have variance TARGET_COV_SCALE.
    """
    if d < N_COMPONENTS:
        raise ValueError(f"The target requires d >= {N_COMPONENTS}; got d={d}.")

    means = torch.zeros((N_COMPONENTS, d), dtype=torch.get_default_dtype())
    for k in range(N_COMPONENTS):
        means[k, k] = alpha

    covs = []
    for k in range(N_COMPONENTS):
        diag = TARGET_COV_SCALE * torch.ones(d, dtype=torch.get_default_dtype())
        diag[k] *= ANISOTROPY_FACTOR
        covs.append(torch.diag(diag))

    return means, torch.stack(covs)


def post_log_dens(pt: torch.Tensor):
    """
    Normalized log-density of the current high-dimensional anisotropic GMM.

    Supports pt of shape (d,) or (..., d).
    """
    d = gmm_means.shape[1]
    log_probs = []

    for mean, prec, logdet, w in zip(
        gmm_means, gmm_precisions, gmm_logdets, gmm_weights
    ):
        diff = pt - mean
        quad = torch.einsum("...i,ij,...j->...", diff, prec, diff)
        lp = torch.log(w) - 0.5 * (quad + logdet + d * LOG_2PI)
        log_probs.append(lp)

    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1)


def setup_gmm(d: int):
    """Set the global target and metric objects for the current dimension."""
    global gmm_weights, gmm_means, gmm_covs, gmm_precisions, gmm_logdets, LOG_2PI
    global kernel_length_scale
    global post_log_dens_grad_val, post_log_dens_grad_val_batch
    global post_log_dens_batch, post_grad_log_dens_batch, cross_entropy_metric

    gmm_weights = torch.ones(N_COMPONENTS, dtype=torch.get_default_dtype()) / N_COMPONENTS
    gmm_means, gmm_covs = make_axis_anisotropic_gmm(d)
    gmm_precisions = torch.linalg.inv(gmm_covs)
    gmm_logdets = torch.linalg.slogdet(gmm_covs).logabsdet
    LOG_2PI = torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.get_default_dtype()))

    # Length scale used by the algorithms.
    kernel_length_scale = effective_kernel_length_scale(d)

    post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
    post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)
    post_log_dens_batch = torch.vmap(post_log_dens)

    # KSD requires the score function x -> grad log pi(x), not the log-density.
    post_grad_log_dens = torch.func.grad(post_log_dens)
    post_grad_log_dens_batch = torch.vmap(post_grad_log_dens)

    cross_entropy_metric = CrossEntropy(
        post_log_dens_batch,
        is_log_dens_vectorized=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def imq_kernel(x: torch.Tensor, y: torch.Tensor, length_scale: float):
    diff = x - y
    sq_dist = (diff * diff).sum(-1)
    c2 = length_scale ** 2
    beta = 0.5
    return (c2 + sq_dist).pow(-beta)


def normalize_wts(wts: torch.Tensor | None):
    if wts is None:
        return None
    return wts / wts.sum()


def make_ksd_metric(kernel_name: str, sigma: float):
    if kernel_name == "RBF":
        return KernelSteinDiscrepancy(
            post_grad_log_dens_batch,
            kernel_length_scale=sigma,
            is_grad_vectorized=True,
        )

    if kernel_name == "IMQ":
        return KernelSteinDiscrepancy(
            post_grad_log_dens_batch,
            kernel_length_scale=sigma,
            kernel_elem=imq_kernel,
            is_grad_vectorized=True,
        )

    raise ValueError(f"Unknown KSD kernel: {kernel_name}")


def metric_names() -> list[str]:
    names = ["CE_unweighted", "CE_weighted"]
    for sigma in SIGMAS:
        for kernel_name in KSD_KERNELS:
            names.append(f"KSD_{kernel_name}_sigma_{sigma}_unweighted")
            names.append(f"KSD_{kernel_name}_sigma_{sigma}_weighted")
    return names


def evaluate_final_metrics(pts: torch.Tensor, wts: torch.Tensor | None = None) -> dict[str, float]:
    """
    Evaluate CE and KSD variants on final particles.

    Weighted metrics use normalized weights when available. For unweighted
    algorithms, weighted and unweighted values coincide.
    """
    wts_norm = normalize_wts(wts)
    out: dict[str, float] = {}

    out["CE_unweighted"] = cross_entropy_metric(pts, wts=None).item()
    if wts_norm is None:
        out["CE_weighted"] = out["CE_unweighted"]
    else:
        out["CE_weighted"] = cross_entropy_metric(pts, wts=wts_norm).item()

    for sigma in SIGMAS:
        for kernel_name in KSD_KERNELS:
            metric = make_ksd_metric(kernel_name, sigma)

            uw_name = f"KSD_{kernel_name}_sigma_{sigma}_unweighted"
            w_name = f"KSD_{kernel_name}_sigma_{sigma}_weighted"

            out[uw_name] = metric(pts, wts=None).item()
            if wts_norm is None:
                out[w_name] = out[uw_name]
            else:
                out[w_name] = metric(pts, wts=wts_norm).item()

    return out


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RUN HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    """
    Simple Gaussian Monte Carlo quadrature rule used by MSIP-QG/GF.
    """
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad), dtype=pts.dtype, device=pts.device) / N_quad
    return pts, wts


def make_init_particles(n_particles: int, run_idx: int, M: int, d: int) -> torch.Tensor:
    """Create a reproducible high-dimensional initialization for one run."""
    torch.manual_seed(base_seed + 1_000_000 * d + 1000 * M + run_idx)

    init_mean = torch.zeros(d, dtype=torch.get_default_dtype())
    init_mean[0] = INIT_ALPHA

    return init_mean + init_std * torch.randn((n_particles, d))


def run_one_algorithm(
    algo_name: str,
    n_particles: int,
    init_particles: torch.Tensor,
    d: int,
):
    """
    Run one algorithm for T iterations.

    Returns
    -------
    pts : torch.Tensor
        Final particles/support points.
    wts : torch.Tensor | None
        Final weights for weighted algorithms, otherwise None.
    """
    if algo_name == "a-SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=d,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            keep_all=True,
            use_quantile_length_scale=0.5,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None

    if algo_name == "SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=d,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None

    if algo_name == "DeepEnsembles":
        trajectories = deepensembles(
            post_log_dens,
            n_particles,
            T,
            dim=d,
            lr=lr / 100,
            init_particles=init_particles.clone(),
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None

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
        return trajectories[-1], None

    if algo_name == "MSIP-Fredholm":
        msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
        trajectories, traj_wts = msip(
            msip_fredholm,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            gradient_decay=gradient_decay,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], traj_wts[-1]

    if algo_name == "MSIP-GI":
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
            kernel_length_scale=kernel_length_scale,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            gradient_decay=gradient_decay,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], traj_wts[-1]

    if algo_name == "MSIP-GF":
        msip_qg = MSIPQuadGradientFree(
            post_log_dens_batch,
            partial(mc_quad_rule, N_quad=N_QUAD_GF, dim=d),
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=d,
            lr=lr_msip,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            gradient_decay=gradient_decay,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], traj_wts[-1]

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVING / REPORTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def cpu_detach(x: torch.Tensor | None):
    if x is None:
        return None
    return x.detach().cpu()


def save_final_state(
    d: int,
    M: int,
    run_idx: int,
    algo_name: str,
    pts: torch.Tensor,
    wts: torch.Tensor | None,
):
    safe_algo = algo_name.replace("/", "_").replace(" ", "_")

    particles_name = f"d_{d}_M_{M}_run_{run_idx:03d}_{safe_algo}_particles.pt"
    weights_name = f"d_{d}_M_{M}_run_{run_idx:03d}_{safe_algo}_weights.pt"
    combined_name = f"d_{d}_M_{M}_run_{run_idx:03d}_{safe_algo}_state.pt"

    torch.save(cpu_detach(pts), PARTICLES_DIR / particles_name)

    if wts is not None:
        torch.save(cpu_detach(normalize_wts(wts)), WEIGHTS_DIR / weights_name)

    torch.save(
        {
            "dataset": EXP_NAME,
            "d": d,
            "M": M,
            "run_idx": run_idx,
            "algo_name": algo_name,
            "particles": cpu_detach(pts),
            "weights": cpu_detach(normalize_wts(wts)) if wts is not None else None,
            "has_weights": wts is not None,
        },
        COMBINED_STATES_DIR / combined_name,
    )


def summarize_results(results_for_metric: dict[tuple[int, int], dict[str, list[float]]]):
    summary_rows = []

    for d in D_VALUES:
        for M in M_values:
            for algo_name in algo_names:
                vals = torch.tensor(
                    results_for_metric[(d, M)][algo_name],
                    dtype=torch.float64,
                )
                mean = vals.mean().item()
                std = vals.std(unbiased=True).item() if R > 1 else 0.0
                summary_rows.append(
                    {
                        "d": d,
                        "M": M,
                        "algo_name": algo_name,
                        "mean": mean,
                        "std": std,
                    }
                )

    return summary_rows


def print_summary_table(metric_name: str, summary_rows: list[dict[str, Any]]):
    print(f"\nFinal {metric_name} over R runs")
    print("smaller is better")
    print()
    print(f"{'d':>5}  {'M':>5}  {'Algorithm':<16}  {'mean':>14}  {'std':>14}")
    print("-" * 70)

    for row in summary_rows:
        print(
            f"{row['d']:5d}  {row['M']:5d}  {row['algo_name']:<16}  "
            f"{row['mean']:14.6f}  {row['std']:14.6f}"
        )


def save_metric_files(
    metric_name: str,
    results_for_metric: dict[tuple[int, int], dict[str, list[float]]],
    summary_rows: list[dict[str, Any]],
):
    metric_safe = metric_name.replace("/", "_").replace(" ", "_")

    pt_path = METRICS_DIR / f"{metric_safe}.pt"
    csv_path = METRICS_DIR / f"{metric_safe}.csv"

    payload = {
        "metric_name": metric_name,
        "T": T,
        "R": R,
        "D_VALUES": D_VALUES,
        "M_values": M_values,
        "algo_names": algo_names,
        "results": results_for_metric,
        "summary_rows": summary_rows,
        "SIGMAS": SIGMAS,
        "KSD_KERNELS": KSD_KERNELS,
        "lr": lr,
        "lr_msip": lr_msip,
        "lr_aldi": lr_aldi,
        "kernel_length_scale_base": kernel_length_scale_base,
        "SCALE_KERNEL_WITH_DIM": SCALE_KERNEL_WITH_DIM,
        "kernel_diag_infl": kernel_diag_infl,
        "gradient_decay": gradient_decay,
        "bounds": bounds,
        "N_COMPONENTS": N_COMPONENTS,
        "MODE_SEPARATION_ALPHA": MODE_SEPARATION_ALPHA,
        "TARGET_COV_SCALE": TARGET_COV_SCALE,
        "ANISOTROPY_FACTOR": ANISOTROPY_FACTOR,
        "INIT_ALPHA": INIT_ALPHA,
        "init_std": init_std,
        "base_seed": base_seed,
    }

    torch.save(payload, pt_path)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric_name", "d", "M", "algo_name", "mean", "std"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({"metric_name": metric_name, **row})

    print(f"Saved metric files: {pt_path} and {csv_path}")


def save_combined_summary(
    all_results: dict[str, dict[tuple[int, int], dict[str, list[float]]]],
    all_summary_rows: dict[str, list[dict[str, Any]]],
):
    combined_pt = SUMMARIES_DIR / "all_metrics_summary.pt"
    combined_csv = SUMMARIES_DIR / "all_metrics_summary.csv"

    torch.save(
        {
            "EXP_NAME": EXP_NAME,
            "T": T,
            "R": R,
            "D_VALUES": D_VALUES,
            "M_values": M_values,
            "algo_names": algo_names,
            "all_results": all_results,
            "all_summary_rows": all_summary_rows,
            "SIGMAS": SIGMAS,
            "KSD_KERNELS": KSD_KERNELS,
        },
        combined_pt,
    )

    with combined_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric_name", "d", "M", "algo_name", "mean", "std"],
        )
        writer.writeheader()
        for metric_name, rows in all_summary_rows.items():
            for row in rows:
                writer.writerow({"metric_name": metric_name, **row})

    print(f"Saved combined summary: {combined_pt} and {combined_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    names = metric_names()

    all_results = {
        metric_name: {
            (d, M): {algo_name: [] for algo_name in algo_names}
            for d in D_VALUES
            for M in M_values
        }
        for metric_name in names
    }

    print("Running CE/KSD benchmark for high-dimensional anisotropic GMMs")
    print(f"D_VALUES = {D_VALUES}")
    print(f"T = {T}, R = {R}, M_values = {M_values}")
    print(f"lr = {lr}, lr_msip = {lr_msip}, lr_aldi = {lr_aldi}")
    print(f"algorithm kernel_length_scale_base = {kernel_length_scale_base}")
    print(f"SCALE_KERNEL_WITH_DIM = {SCALE_KERNEL_WITH_DIM}")
    print(f"evaluation SIGMAS = {SIGMAS}")
    print(f"evaluation KSD_KERNELS = {KSD_KERNELS}")
    print(f"results folder = {ROOT_DIR}")
    print()

    for d in D_VALUES:
        setup_gmm(d)

        print("#" * 72)
        print(f"Dimension d = {d}")
        print(f"algorithm kernel length scale = {kernel_length_scale}")
        print(f"mode separation alpha = {MODE_SEPARATION_ALPHA}")
        print(f"target cov scale = {TARGET_COV_SCALE}")
        print(f"anisotropy factor = {ANISOTROPY_FACTOR}")
        print("#" * 72)

        for M in M_values:
            print(f"=== d = {d}, M = {M} ===")
            for run_idx in range(R):
                init_particles = make_init_particles(M, run_idx, M, d)

                for algo_name in algo_names:
                    pts, wts = run_one_algorithm(algo_name, M, init_particles, d)
                    save_final_state(d, M, run_idx, algo_name, pts, wts)

                    metrics = evaluate_final_metrics(pts, wts=wts)
                    for metric_name, value in metrics.items():
                        all_results[metric_name][(d, M)][algo_name].append(value)

                print(f"  run {run_idx + 1:02d}/{R} done")
            print()

    all_summary_rows = {}

    for metric_name in names:
        summary_rows = summarize_results(all_results[metric_name])
        all_summary_rows[metric_name] = summary_rows
        print_summary_table(metric_name, summary_rows)
        save_metric_files(metric_name, all_results[metric_name], summary_rows)

    save_combined_summary(all_results, all_summary_rows)

    print("\nAll done.")
    print(f"Everything saved under: {ROOT_DIR}")


if __name__ == "__main__":
    main()