from functools import partial

import torch
import matplotlib.pyplot as plt

import nak_torch
from functions import himmelblau
from nak_torch.algorithms import grad_aldi, eks, msip, msip_gs, svgd
from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)
from nak_torch.tools.metrics import CrossEntropy
# from nak_torch.tools.quadrature import (
#     spherical_MC_radial_Laguerre,
#     spherical_struct_radial_Laguerre,
# )
from nak_torch.tools.kernel import kernel_optimal_weight_factory, default_kernel_matrix


# ── Device / dtype ────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_dtype(torch.float64)
torch.manual_seed(314159)


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


# ── Target: Himmelblau density ────────────────────────────────────────────────
# Same target as your previous Himmelblau file.
# The scalar controls concentration around the minima.
post_log_dens = himmelblau(50.0)

post_log_dens_grad_val = torch.func.grad_and_value(post_log_dens)
post_log_dens_grad_val_batch = torch.vmap(post_log_dens_grad_val)

# Vectorized versions for metrics.
post_log_dens_batch = torch.vmap(post_log_dens)
post_log_dens_grad_batch = torch.vmap(torch.func.grad(post_log_dens))

cross_entropy_metric = CrossEntropy(
    post_log_dens_batch,
    is_log_dens_vectorized=True,
)


# ── Gaussian model wrapper needed by EKS ─────────────────────────────────────
# This is kept only to preserve file-1 structure. For Himmelblau, EKS is not a
# natural baseline unless you define a matching inverse-problem Gaussian model.
# We therefore keep the block but do not add EKS to algo_trajs below by default.
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
n_steps = 500
n_particles = 100
lr = 0.2
lr_msip = 0.2

kernel_length_scale = 0.5
kernel_diag_infl = 1e-6
gradient_decay = 1.0
bounds = (-15.0, 15.0)

# Same spirit as your older Himmelblau script: initialize away from the modes.
init_mean = torch.tensor([8.0, 8.0])
init_std = 1.0
init_particles = init_mean + init_std * torch.randn((n_particles, 2))


# ── Quadrature rule ───────────────────────────────────────────────────────────
def mc_quad_rule(batch_size: int, N_quad: int = 40, dim: int = 2):
    pts = torch.randn((batch_size, N_quad, dim))
    wts = torch.ones((batch_size, N_quad)).div_(N_quad)
    return pts, wts


def spherical_quad_(batch_size: int, N_quad: int = 10, dim: int = 2):
    dimension = dim
    N_spherical = 10
    N_radial = int(N_quad / 10)
    pts, wts = spherical_MC_radial_Laguerre(
        batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
    )
    return pts, wts


def spherical_quad(batch_size: int, N_quad: int = 40, dim: int = 2):
    dimension = dim
    N_spherical = 2 * dimension
    N_radial = max(1, int(N_quad / N_spherical))
    pts, wts = spherical_struct_radial_Laguerre(
        batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
    )
    return pts, wts


# ══════════════════════════════════════════════════════════════════════════════
# Run algorithms
# ══════════════════════════════════════════════════════════════════════════════

print("=== SVGD ===")
trajectories_svgd = svgd(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    keep_all=True, compile_step=False, verbose=True,
)

print("=== GI-ALDI ===")
trajectories_galdi = grad_aldi(
    post_log_dens, n_particles, n_steps, dim=2,
    lr=lr / 3, init_particles=init_particles,
    keep_all=True, compile_step=False, verbose=True,
)

# print("=== EKS ===")
# trajectories_eks = eks(
#     model, n_particles=n_particles, n_steps=n_steps, dim=2,
#     lr=lr / 3000, init_particles=init_particles,
#     keep_all=True, compile_step=False, verbose=True,
# )

print("=== MSIP-Fredholm ===")
msip_fredholm = MSIPFredholm(gradient_decay, post_log_dens_grad_val_batch)
trajectories_msip_f, traj_wts_msip_f = msip(
    msip_fredholm, n_particles, n_steps, dim=2,
    lr=lr_msip, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds, gradient_decay=gradient_decay,
    keep_all=True, compile_step=False, verbose=True,
)

# Gradient-free MSIP is the closest Himmelblau analogue of your previous file.
print("=== MSIP-GF ===")
msip_gf = MSIPQuadGradientFree(
    post_log_dens, mc_quad_rule,
)
trajectories_msip_gf, traj_wts_msip_gf = msip(
    msip_gf, n_particles, n_steps, dim=2,
    lr=0.6, init_particles=init_particles,
    kernel_length_scale=kernel_length_scale,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True, compile_step=False, verbose=True,
)

# Optional gradient-informed quadrature MSIP, kept in the same style as file 1.
# print("=== MSIP-QG ===")
# msip_quadgrad = MSIPQuadGradientInformed(
#     post_log_dens_grad_val_batch,
#     partial(spherical_quad, N_quad=40),
#     gradient_decay,
# )
# trajectories_msip_qg, traj_wts_msip_qg = msip(
#     msip_quadgrad, n_particles, n_steps, dim=2,
#     lr=lr_msip, init_particles=init_particles,
#     kernel_length_scale=kernel_length_scale,
#     kernel_diag_infl=kernel_diag_infl,
#     bounds=bounds,
#     keep_all=True, compile_step=False, verbose=True,
# )


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ══════════════════════════════════════════════════════════════════════════════

algo_trajs = {
    "SVGD": trajectories_svgd,
    "GI-ALDI": trajectories_galdi,
    # "EKS": trajectories_eks,
    "MSIP-Fredholm": trajectories_msip_f,
    "MSIP-GF": trajectories_msip_gf,
    # "MSIP-QG": trajectories_msip_qg,
}

algo_few_trajs = {
    "SVGD": trajectories_svgd,
    "GI-ALDI": trajectories_galdi,
}

algo_last_wts = {
    "MSIP-Fredholm": traj_wts_msip_f[-1],
    "MSIP-GF": traj_wts_msip_gf[-1],
    # "MSIP-QG": traj_wts_msip_qg[-1],
}

algo_wts_trajs = {
    "MSIP-Fredholm": traj_wts_msip_f,
    "MSIP-GF": traj_wts_msip_gf,
    # "MSIP-QG": traj_wts_msip_qg,
}

metrics = {
    name: {"ksd_rbf": [], "ksd_imq": [], "cross_entropy": []}
    for name in algo_trajs
}

metric_every = 10
steps_recorded = list(range(0, n_steps, metric_every))


# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# ══════════════════════════════════════════════════════════════════════════════

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

    k_vals = torch.vmap(kernel_fn)(xi_flat, xj_flat)
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


# ══════════════════════════════════════════════════════════════════════════════
# Cross entropy loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing cross entropy …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        pts = traj[step_idx]
        wts = None

        if name in algo_wts_trajs:
            wts = algo_wts_trajs[name][step_idx]
            wts = wts / wts.sum()

        metrics[name]["cross_entropy"].append(
            cross_entropy_metric(pts, wts=wts).item()
        )

print("Cross entropy done.")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (RBF) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_rbf"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, rbf_fn)
        )
print("KSD (RBF) done.")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) loop
# ══════════════════════════════════════════════════════════════════════════════

print("Computing KSD (IMQ) …")
for step_idx in steps_recorded:
    for name, traj in algo_trajs.items():
        metrics[name]["ksd_imq"].append(
            compute_ksd(traj[step_idx], post_log_dens_grad_val_batch, imq_fn)
        )
print("KSD (IMQ) done.")


# ══════════════════════════════════════════════════════════════════════════════
# Contour grid
# ══════════════════════════════════════════════════════════════════════════════

Ngrid = 200
xgrid = torch.linspace(-6, 6, Ngrid)
ygrid = torch.linspace(-6, 6, Ngrid)
X, Y = torch.meshgrid(xgrid, ygrid, indexing="ij")
grid_pts = torch.stack((X.flatten(), Y.flatten()), 1)
Z = post_log_dens(grid_pts).reshape(Ngrid, Ngrid)

z_min = Z.max() - 20  # show 20 log-units below the peak
levels = torch.linspace(z_min, Z.max(), 20).detach().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Individual particle PDFs (one per algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def plot_particles(ax, pts, wts=None, title=""):
    ax.contour(X.cpu(), Y.cpu(), Z.cpu(), levels=levels, alpha=0.6)
    c = torch.abs(wts.detach().cpu()) if wts is not None else None
    sc = ax.scatter(
        pts[:, 0].detach().cpu(), pts[:, 1].detach().cpu(),
        c=c, cmap="viridis", alpha=0.7, s=60,
        edgecolors="k", linewidths=0.4,
    )
    if wts is not None:
        plt.colorbar(sc, ax=ax, shrink=0.8, label="weight")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
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
    fname = (
        f"himmelblau_particles_{name.replace(' ', '_').replace('/', '-')}_"
        f"sigma_{kernel_length_scale}.pdf"
    )
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Cross entropy vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.plot(steps_recorded, metrics[name]["cross_entropy"], label=name)

ax.set_xlabel("Iteration")
ax.set_ylabel(r"Cross entropy  $-\mathbb{E}_{\mu}[\log \pi]$")
ax.set_title("Cross entropy vs Iteration – Himmelblau")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("himmelblau_cross_entropy_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved himmelblau_cross_entropy_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (RBF) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – Himmelblau")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("himmelblau_KSD_RBF_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved himmelblau_KSD_RBF_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")


# ══════════════════════════════════════════════════════════════════════════════
# KSD (IMQ) vs iteration
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – Himmelblau")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("himmelblau_KSD_IMQ_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved himmelblau_KSD_IMQ_all_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")


# ══════════════════════════════════════════════════════════════════════════════
# KSD plots - few algorithms
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_rbf"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (RBF kernel)")
ax.set_title("KSD [RBF] vs Iteration – Himmelblau")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("himmelblau_KSD_RBF_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved himmelblau_KSD_RBF_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")

fig, ax = plt.subplots(figsize=(9, 5))
for name in algo_few_trajs:
    ax.semilogy(steps_recorded, metrics[name]["ksd_imq"], label=name)
ax.set_xlabel("Iteration")
ax.set_ylabel("KSD (IMQ kernel)")
ax.set_title("KSD [IMQ] vs Iteration – Himmelblau")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("himmelblau_KSD_IMQ_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")
plt.close()
print("Saved himmelblau_KSD_IMQ_notall_algorithms_sigma_" + str(kernel_length_scale) + ".pdf")

print("All done.")