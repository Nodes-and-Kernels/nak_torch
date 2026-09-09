from functools import partial
import sys
import os
from pathlib import Path
import csv

import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import nak_torch
from nak_torch.algorithms import cbs, grad_aldi, msip, svgd
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

EXP_NAME = "gmm2d"
PARAM = ""
EXP_NAME = EXP_NAME + "_" + str(PARAM)

# Main output directory. Everything lives under this one EXP_NAME folder.
OUT_DIR = Path("results") / EXP_NAME
METRICS_DIR = OUT_DIR / "metrics"
RAW_DIR = METRICS_DIR / "raw_values"
SUMMARY_DIR = METRICS_DIR / "summary_values"
STATE_DIR = OUT_DIR / "final_states"
PARTICLES_DIR = STATE_DIR / "particles"
WEIGHTS_DIR = STATE_DIR / "weights"
STATE_COMBINED_DIR = STATE_DIR / "combined_particles_weights"
PLOT_DIR = OUT_DIR / "plots" / "pdf"

for d in [
    OUT_DIR,
    METRICS_DIR,
    RAW_DIR,
    SUMMARY_DIR,
    STATE_DIR,
    PARTICLES_DIR,
    WEIGHTS_DIR,
    STATE_COMBINED_DIR,
    PLOT_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ── Benchmark hyper-parameters ────────────────────────────────────────────────
# T iterations, R independent runs, and M particles.
T = 1000
M_values = [25]
R = 5

# Learning rates. Change these here if needed.
lr = 0.5
lr_msip = 0.5
lr_aldi = 0.005

# CBS hyperparameters.
lr_cbs = 0.05
inverse_temp_cbs = 1.0

# Algorithm kernel length scale, used by SVGD/MSIP updates.
# This is separate from the evaluation KSD sigmas below.
kernel_length_scale = 0.5
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-1000.0, 1000.0)

# Metric evaluation choices.
#KSD_SIGMAS = [0.1, 1.0, 10.0]
KSD_SIGMAS = [0.01, 0.05, 0.1, 0.5, 1.0]
KSD_KERNELS = ["RBF", "IMQ"]


# ══════════════════════════════════════════════════════════════════════════════
# Kernels and target
# ══════════════════════════════════════════════════════════════════════════════

def imq_kernel(x: torch.Tensor, y: torch.Tensor, length_scale: float):
    """
    IMQ kernel used inside KernelSteinDiscrepancy.

    k(x,y) = (sigma^2 + ||x-y||^2)^(-1/2)

    The argument name length_scale is kept to match nak_torch's expected
    kernel_elem signature.
    """
    diff = x - y
    sq_dist = (diff * diff).sum(-1)
    c2 = length_scale ** 2
    beta = 0.5
    return (c2 + sq_dist).pow(-beta)


# ── Target: 2D GMM density ───────────────────────────────────────────────────
DIM = 2

# This follows the same target as your original gmm2d script.
# It is an unnormalized Gaussian mixture log-density: mixture weights and
# quadratic forms are included, but Gaussian normalization constants are not.
gmm_weights = torch.tensor([1 / 5] * 5, dtype=torch.get_default_dtype())

gmm_means = torch.stack([
    2 * torch.tensor([ 6.2, -6.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([-4.0,  5.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([ 7.0,  3.0], dtype=torch.get_default_dtype()),
    2 * torch.tensor([-6.5, -4.5], dtype=torch.get_default_dtype()),
    2 * torch.tensor([ 1.0,  7.0], dtype=torch.get_default_dtype()),
])

gmm_covs = torch.stack([
    torch.tensor([[1.5,  0.1], [0.1,  0.5]], dtype=torch.get_default_dtype()),
    torch.tensor([[2.0, -0.6], [-0.6, 0.5]], dtype=torch.get_default_dtype()),
    torch.tensor([[0.7,  0.4], [0.4,  1.2]], dtype=torch.get_default_dtype()),
    torch.tensor([[1.3, -0.5], [-0.5, 0.9]], dtype=torch.get_default_dtype()),
    torch.tensor([[0.6,  0.35], [0.35, 1.6]], dtype=torch.get_default_dtype()),
])

gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt: torch.Tensor):
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j->...", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# Vectorized log-density, used by gradient-free MSIP and CrossEntropy.
post_log_dens_batch = torch.vmap(post_log_dens)

# KSD needs the score function: grad log pi(x).
post_grad_log_dens = torch.func.grad(post_log_dens)
post_grad_log_dens_batch = torch.vmap(post_grad_log_dens)


# Same spirit as your older gmm2d script: initialize from N(0, I).
init_mean = torch.tensor([0.0, 0.0])
init_std = 1.0
base_seed = 314179


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_cross_entropy_metric():
    """
    Build CrossEntropy robustly.

    In your metrics.py, CrossEntropy is a GradFreeMetric and usually accepts
    is_log_dens_vectorized=True. If your local version does not expose that
    keyword, the fallback should still work.
    """
    try:
        return CrossEntropy(post_log_dens_batch, is_log_dens_vectorized=True)
    except TypeError:
        return CrossEntropy(post_log_dens_batch)


cross_entropy_metric = make_cross_entropy_metric()


def make_ksd_metric(kernel_name: str, sigma: float):
    if kernel_name == "RBF":
        return KernelSteinDiscrepancy(
            post_grad_log_dens_batch,
            kernel_length_scale=sigma,
        )
    if kernel_name == "IMQ":
        return KernelSteinDiscrepancy(
            post_grad_log_dens_batch,
            kernel_length_scale=sigma,
            kernel_elem=imq_kernel,
        )
    raise ValueError(f"Unknown KSD kernel: {kernel_name}")


def normalize_weights(wts: torch.Tensor | None):
    if wts is None:
        return None
    return wts / wts.sum()


def evaluate_final_metrics(pts: torch.Tensor, wts: torch.Tensor | None = None):
    """
    Return all requested final metrics for one final particle cloud.

    For algorithms without weights, weighted values are set equal to the
    unweighted values. For MSIP methods, weighted values use normalized MSIP
    weights and unweighted values ignore them.
    """
    wts = normalize_weights(wts)
    out: dict[str, float] = {}

    # Cross entropy: reported only once, independent of KSD kernel/sigma.
    ce_unweighted = cross_entropy_metric(pts, wts=None).item()
    if wts is None:
        ce_weighted = ce_unweighted
    else:
        ce_weighted = cross_entropy_metric(pts, wts=wts).item()

    out["cross_entropy_unweighted"] = ce_unweighted
    out["cross_entropy_weighted"] = ce_weighted

    # KSD: reported for RBF/IMQ and sigma in {0.1, 1.0, 10.0}.
    for sigma in KSD_SIGMAS:
        for kernel_name in KSD_KERNELS:
            metric = make_ksd_metric(kernel_name, sigma)
            key_base = f"ksd_{kernel_name.lower()}_sigma_{sigma:g}"

            ksd_unweighted = metric(pts, wts=None).item()
            if wts is None:
                ksd_weighted = ksd_unweighted
            else:
                ksd_weighted = metric(pts, wts=wts).item()

            out[f"{key_base}_unweighted"] = ksd_unweighted
            out[f"{key_base}_weighted"] = ksd_weighted

    return out


METRIC_NAMES = (
    ["cross_entropy_unweighted", "cross_entropy_weighted"]
    + [
        f"ksd_{kernel.lower()}_sigma_{sigma:g}_{weighting}"
        for sigma in KSD_SIGMAS
        for kernel in KSD_KERNELS
        for weighting in ["unweighted", "weighted"]
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# Single-run helpers
# ══════════════════════════════════════════════════════════════════════════════

#N_QUAD = 1
N_QUAD_GF = 10


def mc_quad_rule(batch_size: int, N_quad: int = 1, dim: int = DIM):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def make_init_particles(n_particles: int, run_idx: int, M: int) -> torch.Tensor:
    """Create a reproducible initialization for one run."""
    torch.manual_seed(base_seed + 1000 * M + run_idx)
    return init_mean + init_std * torch.randn((n_particles, DIM))


def run_one_algorithm(algo_name: str, n_particles: int, init_particles: torch.Tensor):
    """
    Run one algorithm for T iterations.

    Returns
    -------
    pts : Tensor
        Final particles.
    wts : Tensor | None
        Final weights when available. None for unweighted methods.
    """
    if algo_name == "a-SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=DIM,
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
            dim=DIM,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None

    if algo_name == "GA":
        trajectories = deepensembles(
            post_log_dens,
            n_particles,
            T,
            dim=DIM,
            lr=lr / 100,
            init_particles=init_particles.clone(),
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None


    if algo_name == "CBS":
        trajectories = cbs(
            post_log_dens,
            n_particles,
            T,
            dim=DIM,
            lr=lr_cbs,
            inverse_temp=inverse_temp_cbs,
            init_particles=init_particles.clone(),
            bounds=bounds,
            keep_all=True,
            is_log_density_batched=False,
            compile_step=False,
            verbose=False,
        )
        return trajectories[-1], None

    if algo_name == "ALDI":
        trajectories = grad_aldi(
            post_log_dens,
            n_particles,
            T,
            dim=DIM,
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
            dim=DIM,
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

    if algo_name == "MSIP-GI-1":
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=1),
            1.0,
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=DIM,
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
    
    if algo_name == "MSIP-GI-10":
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=10),
            1.0,
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=DIM,
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
            partial(mc_quad_rule, N_quad=N_QUAD_GF),
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=DIM,
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
# Saving helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_name(name: str) -> str:
    """Make algorithm/metric names safe for filenames."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(".", "p")
    )


def metric_filename(metric_name: str, suffix: str):
    return f"{EXP_NAME}_{metric_name}_T_{T}_R_{R}_algos_{len(algo_names)}.{suffix}"


def save_final_state(algo_name: str, M: int, run_idx: int, pts: torch.Tensor, wts: torch.Tensor | None):
    """
    Save final particles and final weights for one algorithm/run.

    Files created:
      final_states/particles/<algo>/...pt
      final_states/particles/<algo>/...csv
      final_states/weights/<algo>/...pt
      final_states/weights/<algo>/...csv
      final_states/combined_particles_weights/<algo>/...pt

    For unweighted methods, weights are saved as uniform weights. This makes
    downstream comparison easier because every algorithm has a weight file.
    """
    algo_dir_name = safe_name(algo_name)
    particles_algo_dir = PARTICLES_DIR / algo_dir_name
    weights_algo_dir = WEIGHTS_DIR / algo_dir_name
    combined_algo_dir = STATE_COMBINED_DIR / algo_dir_name
    for d in [particles_algo_dir, weights_algo_dir, combined_algo_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pts_cpu = pts.detach().cpu()
    if wts is None:
        wts_cpu = torch.ones(pts_cpu.shape[0], dtype=pts_cpu.dtype) / pts_cpu.shape[0]
        has_algorithm_weights = False
    else:
        wts_cpu = normalize_weights(wts.detach()).cpu()
        has_algorithm_weights = True

    stem = f"{EXP_NAME}_{algo_dir_name}_M_{M}_run_{run_idx:03d}_T_{T}"

    particles_pt = particles_algo_dir / f"{stem}_particles.pt"
    particles_csv = particles_algo_dir / f"{stem}_particles.csv"
    weights_pt = weights_algo_dir / f"{stem}_weights.pt"
    weights_csv = weights_algo_dir / f"{stem}_weights.csv"
    combined_pt = combined_algo_dir / f"{stem}_particles_weights.pt"

    torch.save(pts_cpu, particles_pt)
    torch.save(wts_cpu, weights_pt)
    torch.save(
        {
            "algorithm": algo_name,
            "M": M,
            "run_idx": run_idx,
            "T": T,
            "particles": pts_cpu,
            "weights": wts_cpu,
            "has_algorithm_weights": has_algorithm_weights,
            "note": "For unweighted algorithms, weights are uniform.",
        },
        combined_pt,
    )

    with particles_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_idx"] + [f"x{j}" for j in range(pts_cpu.shape[1])])
        for i, row in enumerate(pts_cpu.tolist()):
            writer.writerow([i] + row)

    with weights_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_idx", "weight", "has_algorithm_weights"])
        for i, value in enumerate(wts_cpu.tolist()):
            writer.writerow([i, value, has_algorithm_weights])

    return {
        "algorithm": algo_name,
        "M": M,
        "run_idx": run_idx,
        "particles_pt": str(particles_pt),
        "particles_csv": str(particles_csv),
        "weights_pt": str(weights_pt),
        "weights_csv": str(weights_csv),
        "combined_pt": str(combined_pt),
        "has_algorithm_weights": has_algorithm_weights,
    }


def save_metric_separate_files(metric_name: str, results: dict, summary_rows: list[dict]):
    """
    Save one metric into separate files:
      - metrics/raw_values/<metric>.pt
      - metrics/raw_values/<metric>.csv
      - metrics/summary_values/<metric>_summary.pt
      - metrics/summary_values/<metric>_summary.csv
    """
    raw_payload = {
        "metric_name": metric_name,
        "values": {
            M: {algo: results[M][algo][metric_name] for algo in algo_names}
            for M in M_values
        },
        "T": T,
        "R": R,
        "M_values": M_values,
        "algo_names": algo_names,
        "KSD_SIGMAS": KSD_SIGMAS,
        "KSD_KERNELS": KSD_KERNELS,
        "algorithm_kernel_length_scale": kernel_length_scale,
        "kernel_diag_infl": kernel_diag_infl,
        "gradient_decay": gradient_decay,
        "bounds": bounds,
        "lr": lr,
        "lr_msip": lr_msip,
        "lr_aldi": lr_aldi,
        "lr_cbs": lr_cbs,
        "inverse_temp_cbs": inverse_temp_cbs,
        "DIM": DIM,
        "base_seed": base_seed,
    }

    raw_pt_path = RAW_DIR / metric_filename(metric_name, "pt")
    raw_csv_path = RAW_DIR / metric_filename(metric_name, "csv")
    summary_pt_path = SUMMARY_DIR / metric_filename(metric_name + "_summary", "pt")
    summary_csv_path = SUMMARY_DIR / metric_filename(metric_name + "_summary", "csv")

    torch.save(raw_payload, raw_pt_path)

    with raw_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "M", "algorithm", "run_idx", "value"])
        for M in M_values:
            for algo in algo_names:
                for run_idx, value in enumerate(results[M][algo][metric_name]):
                    writer.writerow([metric_name, M, algo, run_idx, value])

    metric_summary_rows = [row for row in summary_rows if row["metric"] == metric_name]
    torch.save({"metric_name": metric_name, "summary_rows": metric_summary_rows}, summary_pt_path)

    with summary_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "M", "algorithm", "mean", "std"])
        writer.writeheader()
        writer.writerows(metric_summary_rows)

    return raw_pt_path, raw_csv_path, summary_pt_path, summary_csv_path


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

algo_names = [
    "a-SVGD",
    "SVGD",
    "ALDI",
    "MSIP-Fredholm",
    "MSIP-GI-1",
    "MSIP-GI-10",
    "GA",
    "MSIP-GF",
    "CBS",
]

results = {
    M: {
        algo_name: {metric_name: [] for metric_name in METRIC_NAMES}
        for algo_name in algo_names
    }
    for M in M_values
}

# Bookkeeping for saved particles/weights files.
saved_state_rows: list[dict] = []

print("Running final CE/KSD benchmark")
print(f"T = {T}, R = {R}, M_values = {M_values}")
print(f"Algorithm kernel_length_scale = {kernel_length_scale}")
print(f"CBS lr = {lr_cbs}, inverse_temp = {inverse_temp_cbs}")
print(f"Evaluation KSD sigmas = {KSD_SIGMAS}")
print(f"Evaluation KSD kernels = {KSD_KERNELS}")
print()

for M in M_values:
    print(f"=== M = {M} ===")
    for run_idx in range(R):
        init_particles = make_init_particles(M, run_idx, M)

        for algo_name in algo_names:
            pts, wts = run_one_algorithm(algo_name, M, init_particles)

            saved_state_rows.append(
                save_final_state(algo_name, M, run_idx, pts=pts, wts=wts)
            )

            metrics = evaluate_final_metrics(pts, wts=wts)

            for metric_name in METRIC_NAMES:
                results[M][algo_name][metric_name].append(metrics[metric_name])

        print(f"  run {run_idx + 1:02d}/{R} done")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Report tables and save files
# ══════════════════════════════════════════════════════════════════════════════

summary_rows: list[dict] = []

for metric_name in METRIC_NAMES:
    print(f"\nFinal {metric_name} over R runs")
    print("smaller is better")
    print()
    print(f"{'M':>5}  {'Algorithm':<16}  {'mean':>14}  {'std':>14}")
    print("-" * 55)

    for M in M_values:
        for algo_name in algo_names:
            vals = torch.tensor(results[M][algo_name][metric_name], dtype=torch.float64)
            mean = vals.mean().item()
            std = vals.std(unbiased=True).item() if R > 1 else 0.0
            row = {
                "metric": metric_name,
                "M": M,
                "algorithm": algo_name,
                "mean": mean,
                "std": std,
            }
            summary_rows.append(row)
            print(f"{M:5d}  {algo_name:<16}  {mean:14.6f}  {std:14.6f}")


# Save combined raw and summary payloads.
combined_out = OUT_DIR / f"{EXP_NAME}_all_metrics_T_{T}_R_{R}.pt"
combined_summary_csv = OUT_DIR / f"{EXP_NAME}_all_metrics_summary_T_{T}_R_{R}.csv"
combined_raw_csv = OUT_DIR / f"{EXP_NAME}_all_metrics_raw_T_{T}_R_{R}.csv"

torch.save(
    {
        "T": T,
        "R": R,
        "M_values": M_values,
        "algo_names": algo_names,
        "metric_names": METRIC_NAMES,
        "results": results,
        "summary_rows": summary_rows,
        "saved_state_rows": saved_state_rows,
        "output_directories": {
            "out_dir": str(OUT_DIR),
            "metrics_dir": str(METRICS_DIR),
            "raw_metric_values": str(RAW_DIR),
            "summary_metric_values": str(SUMMARY_DIR),
            "final_states": str(STATE_DIR),
            "particles": str(PARTICLES_DIR),
            "weights": str(WEIGHTS_DIR),
            "combined_particles_weights": str(STATE_COMBINED_DIR),
            "plots_pdf": str(PLOT_DIR),
        },
        "lr": lr,
        "lr_msip": lr_msip,
        "lr_aldi": lr_aldi,
        "algorithm_kernel_length_scale": kernel_length_scale,
        "evaluation_ksd_sigmas": KSD_SIGMAS,
        "evaluation_ksd_kernels": KSD_KERNELS,
        "kernel_diag_infl": kernel_diag_infl,
        "gradient_decay": gradient_decay,
        "bounds": bounds,
        "init_mean": init_mean.detach().cpu(),
        "init_std": init_std,
        "base_seed": base_seed,
        "DIM": DIM,
        "gmm_weights": gmm_weights.detach().cpu(),
        "gmm_means": gmm_means.detach().cpu(),
        "gmm_covs": gmm_covs.detach().cpu(),
    },
    combined_out,
)

with combined_summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["metric", "M", "algorithm", "mean", "std"])
    writer.writeheader()
    writer.writerows(summary_rows)

with combined_raw_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "M", "algorithm", "run_idx", "value"])
    for metric_name in METRIC_NAMES:
        for M in M_values:
            for algo_name in algo_names:
                for run_idx, value in enumerate(results[M][algo_name][metric_name]):
                    writer.writerow([metric_name, M, algo_name, run_idx, value])

state_index_csv = STATE_DIR / f"{EXP_NAME}_saved_particles_weights_index_T_{T}_R_{R}.csv"
state_index_pt = STATE_DIR / f"{EXP_NAME}_saved_particles_weights_index_T_{T}_R_{R}.pt"
torch.save(saved_state_rows, state_index_pt)
with state_index_csv.open("w", newline="") as f:
    fieldnames = [
        "algorithm",
        "M",
        "run_idx",
        "particles_pt",
        "particles_csv",
        "weights_pt",
        "weights_csv",
        "combined_pt",
        "has_algorithm_weights",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(saved_state_rows)

print(f"\nSaved combined raw .pt to {combined_out}")
print(f"Saved combined raw .csv to {combined_raw_csv}")
print(f"Saved combined summary .csv to {combined_summary_csv}")
print(f"Saved particles/weights index .pt to {state_index_pt}")
print(f"Saved particles/weights index .csv to {state_index_csv}")

# Save every metric in its own separate files.
print("\nSaving separate files for each metric...")
for metric_name in METRIC_NAMES:
    raw_pt, raw_csv, summary_pt, summary_csv = save_metric_separate_files(
        metric_name, results, summary_rows
    )
    print(f"  {metric_name}")
    print(f"    raw pt:      {raw_pt}")
    print(f"    raw csv:     {raw_csv}")
    print(f"    summary pt:  {summary_pt}")
    print(f"    summary csv: {summary_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Optional: plots, one PDF per metric
# ══════════════════════════════════════════════════════════════════════════════

for metric_name in METRIC_NAMES:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo_name in algo_names:
        means = []
        stds = []
        for M in M_values:
            vals = torch.tensor(results[M][algo_name][metric_name], dtype=torch.float64)
            means.append(vals.mean().item())
            stds.append(vals.std(unbiased=True).item() if R > 1 else 0.0)

        ax.errorbar(M_values, means, yerr=stds, marker="o", capsize=4, label=algo_name)

    ax.set_xlabel("Number of particles M")
    ax.set_ylabel(f"Final {metric_name}")
    ax.set_title(f"Final {metric_name} after T={T} iterations, R={R} runs")
    ax.legend(fontsize=8)
    plt.tight_layout()

    fig_name = PLOT_DIR / f"{EXP_NAME}_{metric_name}_vs_M_T_{T}_R_{R}.pdf"
    plt.savefig(fig_name)
    plt.close()
    print(f"Saved plot: {fig_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Optional: visualize final particle positions for the last run
# ══════════════════════════════════════════════════════════════════════════════

try:
    import numpy as np

    M = M_values[-1]
    run_idx = R - 1
    init_particles = make_init_particles(M, run_idx, M)

    grid_res = 200
    x_range = torch.linspace(-17, 17, grid_res)
    y_range = torch.linspace(-17, 17, grid_res)
    XX, YY = torch.meshgrid(x_range, y_range, indexing="ij")
    grid_pts = torch.stack([XX.flatten(), YY.flatten()], dim=1)
    log_dens_grid = post_log_dens_batch(grid_pts).reshape(grid_res, grid_res)
    dens_grid = torch.exp(log_dens_grid - log_dens_grid.max()).detach().cpu().numpy()

    n_algos = len(algo_names)
    ncols = 3
    nrows = (n_algos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax_idx, algo_name in enumerate(algo_names):
        ax = axes[ax_idx]

        # Light visualization palette
        ax.contourf(
            x_range.detach().cpu().numpy(),
            y_range.detach().cpu().numpy(),
            dens_grid.T,
            levels=35,
            cmap="Greys",
            alpha=0.24,
            zorder=0,
        )

        ax.contour(
            x_range.detach().cpu().numpy(),
            y_range.detach().cpu().numpy(),
            dens_grid.T,
            levels=12,
            colors="grey",
            linewidths=0.7,
            alpha=0.7,
            zorder=1,
        )

        ax.set_facecolor("white")

        pts, wts = run_one_algorithm(algo_name, M, init_particles)
        xy = pts.detach().cpu().numpy()

        if wts is not None:
            wts_np = normalize_weights(wts).detach().cpu().numpy()
            sizes = 70 * wts_np / wts_np.max()

            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=sizes,
                c="#c1121f",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=35,
                c="#c1121f",
                edgecolors="white",
                linewidths=0.5,
                alpha=0.95,
                zorder=3,
            )

        ax.set_xlim(-17, 17)
        ax.set_ylim(-17, 17)
        ax.set_title(algo_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    for ax in axes[n_algos:]:
        ax.set_visible(False)

    fig.suptitle(f"Final particles — GMM 2D (M={M}, T={T}, run {R}/{R})", fontsize=13, y=1.02)
    plt.tight_layout()

    particle_fig_name = PLOT_DIR / f"{EXP_NAME}_final_particles_M_{M}_T_{T}_run_{R}.pdf"
    plt.savefig(particle_fig_name)
    plt.close()
    print(f"Saved particle visualization: {particle_fig_name}")
except Exception as exc:
    print(f"Skipped final particle visualization because of: {exc}")

print("All done.")
