from functools import partial
import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import nak_torch
from nak_torch.algorithms import grad_aldi, msip, svgd
from nak_torch.algorithms.deepensembles import deepensembles
from nak_torch.algorithms.msip import MSIPFredholm,MSIPQuadGradientInformed
from functions import himmelblau
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

EXP_NAME = 'himmelblau'
PARAM = '1'
EXP_NAME = EXP_NAME +'_'+ str(PARAM)

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("results/"+EXP_NAME)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("results/"+EXP_NAME+"/pdf")
OUT_DIR.mkdir(parents=True, exist_ok=True)





# ── Target: Himmelblau density ────────────────────────────────────────────────

beta = float(PARAM)
post_log_dens = himmelblau(beta)

post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# Vectorized version for the metric.
post_log_dens_batch = torch.vmap(post_log_dens)

cross_entropy_metric = CrossEntropy(
    post_log_dens_batch,
    is_log_dens_vectorized=True,
)


# ── Benchmark hyper-parameters ────────────────────────────────────────────────
# T iterations, R independent runs, and M particles.
T = 1000
M_values = [25]
R = 20


# Learning rates. Change these here if needed.
lr = 0.1
lr_msip = 0.1
lr_aldi = 0.005

kernel_length_scale = 0.1
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-1000.0, 1000.0)

# Same spirit as your older Himmelblau script: initialize away from the modes.
init_mean = torch.tensor([8.0, 8.0])
init_std = 1.0

base_seed = 314159


# ══════════════════════════════════════════════════════════════════════════════
# Single-run helpers
# ══════════════════════════════════════════════════════════════════════════════

N_QUAD = 1
def mc_quad_rule(batch_size: int, N_quad: int = N_QUAD, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts

def make_init_particles(n_particles: int, run_idx: int, M: int) -> torch.Tensor:
    """Create a reproducible initialization for one run."""
    torch.manual_seed(base_seed + 1000 * M + run_idx)
    return init_mean + init_std * torch.randn((n_particles, 2))

# def make_init_particles(n_particles: int, run_idx: int, M: int) -> torch.Tensor:
#     """Create a reproducible initialization for one run."""
#     torch.manual_seed(base_seed + 1000 * M + run_idx)
#     return -5.0 + 10.0 * torch.rand((n_particles, 2))


def final_cross_entropy(name: str, pts: torch.Tensor, wts: torch.Tensor | None = None) -> float:
    """Compute final cross entropy; normalize weights when present."""
    if wts is not None:
        wts = wts / wts.sum()
    return cross_entropy_metric(pts, wts=wts).item()


def run_one_algorithm(algo_name: str, n_particles: int, init_particles: torch.Tensor):
    """
    Run one algorithm for T iterations and return only the final cross entropy.
    """
    if algo_name == "a-SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=2,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            keep_all=True,
            use_quantile_length_scale = 0.5,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(algo_name, trajectories[-1])
    if algo_name == "SVGD":
        trajectories = svgd(
            post_log_dens,
            n_particles,
            T,
            dim=2,
            lr=lr,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(algo_name, trajectories[-1])

    if algo_name == "DeepEnsembles":
        trajectories = deepensembles(
            post_log_dens,
            n_particles,
            T,
            dim=2,
            lr=lr/100,
            init_particles=init_particles.clone(),
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        #print(trajectories)
        return final_cross_entropy(algo_name, trajectories[-1])

    if algo_name == "GI-ALDI":
        trajectories = grad_aldi(
            post_log_dens,
            n_particles,
            T,
            dim=2,
            lr=lr_aldi,
            init_particles=init_particles.clone(),
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        return final_cross_entropy(algo_name, trajectories[-1])

    if algo_name == "MSIP-Fredholm":
        msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
        trajectories, traj_wts = msip(
            msip_fredholm,
            n_particles,
            T,
            dim=2,
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
        return final_cross_entropy(algo_name, trajectories[-1], wts=traj_wts[-1])
    
    if algo_name == "MSIP-GI":
        #msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=N_QUAD),
            1.0,
        )
        trajectories, traj_wts = msip(
            msip_qg,
            n_particles,
            T,
            dim=2,
            lr=lr_msip/5,
            init_particles=init_particles.clone(),
            kernel_length_scale=kernel_length_scale,
            kernel_diag_infl=kernel_diag_infl,
            bounds=bounds,
            gradient_decay=gradient_decay,
            keep_all=True,
            compile_step=False,
            verbose=False,
        )
        


        
        return final_cross_entropy(algo_name, trajectories[-1], wts=traj_wts[-1])

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

algo_names = [
    "a-SVGD",
    "SVGD",
    "GI-ALDI",
    "MSIP-Fredholm",
    "MSIP-GI",
    "DeepEnsembles",
]

# results[M][algo] = list of R final cross entropy values
results = {
    M: {algo_name: [] for algo_name in algo_names}
    for M in M_values
}

print("Running final-cross-entropy benchmark")
print(f"T = {T}, R = {R}, M_values = {M_values}")
print(f"lr = {lr}, lr_msip = {lr_msip}, kernel_length_scale = {kernel_length_scale}")
print()

for M in M_values:
    print(f"=== M = {M} ===")
    for run_idx in range(R):
        init_particles = make_init_particles(M, run_idx, M)

        for algo_name in algo_names:
            ce = run_one_algorithm(algo_name, M, init_particles)
            results[M][algo_name].append(ce)

        print(f"  run {run_idx + 1:02d}/{R} done")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Report table
# ══════════════════════════════════════════════════════════════════════════════

print("\nFinal cross entropy over R runs")
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
        summary_rows.append((M, algo_name, mean, std))
        print(f"{M:5d}  {algo_name:<16}  {mean:14.6f}  {std:14.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# Optional: save raw and summary results
# ══════════════════════════════════════════════════════════════════════════════
path = "results/"+EXP_NAME+"/pdf/"

out_name = (
    path+EXP_NAME+"_final_cross_entropy_"
    f"M_{M}_T_{T}_R_{R}_sigma_{kernel_length_scale}.pt"
)

torch.save(
    {
        "T": T,
        "R": R,
        "M_values": M_values,
        "algo_names": algo_names,
        "results": results,
        "summary_rows": summary_rows,
        "lr": lr,
        "lr_msip": lr_msip,
        "kernel_length_scale": kernel_length_scale,
        "kernel_diag_infl": kernel_diag_infl,
        "gradient_decay": gradient_decay,
        "bounds": bounds,
        "init_mean": init_mean.detach().cpu(),
        "init_std": init_std,
        "base_seed": base_seed,
    },
    out_name,
)
print(f"\nSaved raw results to {out_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Optional: plot mean final cross entropy vs M
# ══════════════════════════════════════════════════════════════════════════════

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
ax.set_title(f"Final cross entropy after T={T} iterations, R={R} runs")
ax.legend(fontsize=8)
plt.tight_layout()


fig_name = (
    path+EXP_NAME+"_final_cross_entropy_vs_"
    f"M_{M}_T_{T}_R_{R}_sigma_{kernel_length_scale}.pdf"
)
plt.savefig(fig_name)
plt.close()
print(f"Saved {fig_name}")

# ══════════════════════════════════════════════════════════════════════════════
# Visualize final particle positions (last run only, no saving)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np

# Re-run last run for each algo and collect final particles + weights
M = M_values[-1]
run_idx = R - 1
init_particles = make_init_particles(M, run_idx, M)

# Background grid for the Himmelblau density
grid_res = 200
x_range = torch.linspace(-5, 5, grid_res)
y_range = torch.linspace(-5, 5, grid_res)
XX, YY = torch.meshgrid(x_range, y_range, indexing='ij')
grid_pts = torch.stack([XX.flatten(), YY.flatten()], dim=1)
log_dens_grid = post_log_dens_batch(grid_pts).reshape(grid_res, grid_res)
dens_grid = torch.exp(log_dens_grid - log_dens_grid.max()).cpu().numpy()

n_algos = len(algo_names)
ncols = 3
nrows = (n_algos + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
axes = axes.flatten()

for ax_idx, algo_name in enumerate(algo_names):
    ax = axes[ax_idx]

    # ── draw density background ───────────────────────────────────────────
    ax.contourf(
        x_range.cpu().numpy(), y_range.cpu().numpy(),
        dens_grid.T,          # transpose: contourf expects (ny, nx)
        levels=20, cmap="Blues", alpha=0.6,
    )

    # ── run algorithm, collect final particles & weights ──────────────────
    pts = wts = None

    if algo_name == "a-SVGD":
        trajs = svgd(post_log_dens, M, T, dim=2, lr=lr,
                     init_particles=init_particles.clone(),
                     kernel_length_scale=kernel_length_scale,
                     keep_all=True, use_quantile_length_scale=0.5,
                     compile_step=False, verbose=False)
        pts = trajs[-1]

    elif algo_name == "SVGD":
        trajs = svgd(post_log_dens, M, T, dim=2, lr=lr,
                     init_particles=init_particles.clone(),
                     kernel_length_scale=kernel_length_scale,
                     keep_all=True, compile_step=False, verbose=False)
        pts = trajs[-1]

    elif algo_name == "DeepEnsembles":
        trajs = deepensembles(post_log_dens, M, T, dim=2, lr=lr/100,
                              init_particles=init_particles.clone(),
                              keep_all=True, compile_step=False, verbose=False)
        pts = trajs[-1]

    elif algo_name == "GI-ALDI":
        trajs = grad_aldi(post_log_dens, M, T, dim=2, lr=lr_aldi,
                          init_particles=init_particles.clone(),
                          keep_all=True, compile_step=False, verbose=False)
        pts = trajs[-1]

    elif algo_name == "MSIP-Fredholm":
        msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
        trajs, traj_wts = msip(msip_fredholm, M, T, dim=2, lr=lr_msip,
                               init_particles=init_particles.clone(),
                               kernel_length_scale=kernel_length_scale,
                               kernel_diag_infl=kernel_diag_infl, bounds=bounds,
                               gradient_decay=gradient_decay, keep_all=True,
                               compile_step=False, verbose=False)
        pts = trajs[-1]
        wts = traj_wts[-1]
        wts = (wts / wts.sum()).cpu().numpy()

    elif algo_name == "MSIP-GI":
        msip_qg = MSIPQuadGradientInformed(
            post_log_dens_grad_val_batch,
            partial(mc_quad_rule, N_quad=N_QUAD), 1.0,
        )
        trajs, traj_wts = msip(msip_qg, M, T, dim=2, lr=lr_msip/5,
                               init_particles=init_particles.clone(),
                               kernel_length_scale=kernel_length_scale,
                               kernel_diag_infl=kernel_diag_infl, bounds=bounds,
                               gradient_decay=gradient_decay, keep_all=True,
                               compile_step=False, verbose=False)
        pts = trajs[-1]
        wts = traj_wts[-1]
        wts = (wts / wts.sum()).cpu().numpy()

    # ── scatter particles ─────────────────────────────────────────────────
    xy = pts.cpu().numpy()
    
    if wts is not None:
        # size proportional to weight, clipped for readability
        sizes = np.clip(wts * M * 200, 10, 300)
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c="crimson",
                   alpha=0.8, zorder=3, linewidths=0.4, edgecolors="white")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=40, c="crimson",
                   alpha=0.8, zorder=3, linewidths=0.4, edgecolors="white")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(algo_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")

# hide unused axes
for ax in axes[n_algos:]:
    ax.set_visible(False)

fig.suptitle(
    f"Final particles — Himmelblau (β={beta}, M={M}, T={T}, run {R}/{R})",
    fontsize=13, y=1.02,
)
plt.tight_layout()
plt.show()

print("All done.")
