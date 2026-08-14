from functools import partial

import torch
import matplotlib.pyplot as plt

import nak_torch
from nak_torch.algorithms import grad_aldi, msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm
from nak_torch.tools.metrics import CrossEntropy


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)


# ── Target: 5-component GMM ───────────────────────────────────────────────────
gmm_weights = torch.tensor([1 / 5] * 5)

gmm_means = torch.stack([
    2 * torch.tensor([ 6.2, -6.0]),
    2 * torch.tensor([-4.0,  5.0]),
    2 * torch.tensor([ 7.0,  3.0]),
    2 * torch.tensor([-6.5, -4.5]),
    2 * torch.tensor([ 1.0,  7.0]),
])

gmm_covs = torch.stack([
    torch.tensor([[1.5,  0.1],
                  [0.1,  0.5]]),

    torch.tensor([[2.0, -0.6],
                  [-0.6, 0.5]]),

    torch.tensor([[0.7,  0.4],
                  [0.4,  1.2]]),

    torch.tensor([[1.3, -0.5],
                  [-0.5, 0.9]]),

    torch.tensor([[0.6,  0.35],
                  [0.35, 1.6]]),
])

gmm_precisions = torch.linalg.inv(gmm_covs)


def post_log_dens(pt):
    """
    Unnormalized GMM log-density.

    This matches your current file: it includes mixture weights and the quadratic
    forms, but not the Gaussian normalization constants. That is fine for the
    cross-entropy diagnostic as long as all algorithms use the same target.
    """
    log_probs = []
    for mean, prec, w in zip(gmm_means, gmm_precisions, gmm_weights):
        diff = pt - mean
        lp = torch.log(w) - 0.5 * torch.einsum("...i,ij,...j", diff, prec, diff)
        log_probs.append(lp)
    return torch.stack(log_probs, dim=-1).logsumexp(dim=-1).squeeze()


post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# Vectorized version for the metric. This is faster than looping point-by-point.
post_log_dens_batch = torch.vmap(post_log_dens)

cross_entropy_metric = CrossEntropy(
    post_log_dens_batch,
    is_log_dens_vectorized=True,
)


# ── Benchmark hyper-parameters ────────────────────────────────────────────────
# T iterations, R independent runs, and M particles.
T = 500
R = 20
M_values = [5, 10, 15, 20, 50,100]

# Learning rates. Change these here if needed.
lr = 0.2
lr_msip = 0.2

kernel_length_scale = 0.5
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-100.0, 100.0)

# Same initialization as your GMM file: start away from the modes.
init_mean = torch.tensor([15.0, 15.0])
init_std = 1.0

base_seed = 314159


# ══════════════════════════════════════════════════════════════════════════════
# Single-run helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_init_particles(n_particles: int, run_idx: int, M: int) -> torch.Tensor:
    """Create a reproducible initialization for one run."""
    torch.manual_seed(base_seed + 1000 * M + run_idx)
    return init_mean + init_std * torch.randn((n_particles, 2))


def final_cross_entropy(pts: torch.Tensor, wts: torch.Tensor | None = None) -> float:
    """Compute final cross entropy; normalize weights when present."""
    if wts is not None:
        wts = wts / wts.sum()
    return cross_entropy_metric(pts, wts=wts).item()


def run_one_algorithm(algo_name: str, n_particles: int, init_particles: torch.Tensor) -> float:
    """
    Run one algorithm for T iterations and return only the final cross entropy.
    """
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
        return final_cross_entropy(trajectories[-1])

    if algo_name == "GI-ALDI":
        trajectories = grad_aldi(
            post_log_dens,
            n_particles,
            T,
            dim=2,
            lr=lr / 3,
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
        return final_cross_entropy(trajectories[-1], wts=traj_wts[-1])

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ══════════════════════════════════════════════════════════════════════════════

algo_names = [
    "SVGD",
    "GI-ALDI",
    "MSIP-Fredholm",
]

# results[M][algo] = list of R final cross entropy values
results = {
    M: {algo_name: [] for algo_name in algo_names}
    for M in M_values
}

print("Running GMM final-cross-entropy benchmark")
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

out_name = (
    "gmm_final_cross_entropy_"
    f"T_{T}_R_{R}_sigma_{kernel_length_scale}.pt"
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
        "gmm_weights": gmm_weights.detach().cpu(),
        "gmm_means": gmm_means.detach().cpu(),
        "gmm_covs": gmm_covs.detach().cpu(),
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
ax.set_title(f"GMM final cross entropy after T={T} iterations, R={R} runs")
ax.legend(fontsize=8)
plt.tight_layout()

fig_name = (
    "gmm_final_cross_entropy_vs_M_"
    f"T_{T}_R_{R}_sigma_{kernel_length_scale}.pdf"
)
plt.savefig(fig_name)
plt.close()
print(f"Saved {fig_name}")

print("All done.")