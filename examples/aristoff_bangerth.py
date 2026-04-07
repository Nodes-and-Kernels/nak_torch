# %%
import math
import torch
from functions import aristoff_bangerth as ab, build_aristoff_bangerth
from nak_torch.algorithms import msip, svgd
from matplotlib import ticker
import gc
import matplotlib.pyplot as plt
from nak_torch.tools.kernel import sqexp_kernel_matrix
from tqdm import tqdm
import pandas as pd

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

# %%
log_p = build_aristoff_bangerth(use_compiled=True, dtype=torch.float64)
log_th = torch.randn(500, 64, requires_grad=True, dtype=torch.float64)
test_out = log_p(log_th)
test_eval = torch.autograd.grad(test_out.sum(), log_th)

# %%
del log_th
del test_out
del test_eval
gc.collect()

# %%
n_particles, n_steps, dim = 500, 25, 64
kernel_bandwidth = 0.75

torch.manual_seed(1)
init_particles = 2 * torch.randn(
    (n_particles, dim),
    dtype=torch.float64,
)  # Sample from prior

msip_args = {
    "n_particles": n_particles,
    "n_steps": n_steps,  # "epochs" (passes over all particles)
    "dim": 64,
    "bounds": (-8, 8),   # [a,b]^d
    "gradient_informed": True,
    "lr": 1e-1,
    "noise": 0.05,          # currently unused
    "init_particles": init_particles,
    "kernel_bandwidth": kernel_bandwidth,
    "bandwidth_factor": 0.25,
    "seed": 0,
    "kernel_diag_infl": 1e-10,
    "keep_all": False,
    "device": None
}

# %%
trajectories_msip = msip(
    log_p,
    **msip_args
)

# %%
trajectories_svgd = svgd(
    log_p,
    is_log_density_batched=True,
    **msip_args
)

# %%
side_len = min(6, int(math.floor(math.sqrt(n_particles))))
pts = trajectories_msip[-1][:side_len**2].detach().cpu()# - init_particles[:side_len**2]
fig = plt.figure(figsize=(9, 6), layout='constrained')
gs = fig.add_gridspec(side_len, side_len + 2)
vabs = max(pts.min().abs(), pts.max().abs())
plt_kwargs = {'vmin': -vabs, 'vmax': vabs, 'extent': (0, 8, 0, 8)}

for i in range(side_len):
    for j in range(side_len):
        ax = fig.add_subplot(gs[i, j])
        # ax.set_axis_off()
        ax.set_aspect('equal')
        t = ax.matshow(pts[i*side_len + j].reshape(8, 8), **plt_kwargs)
        # ax.vlines(jnp.arange(1,8), -0.1, 8.1, color='w', lw=0.75)
        # ax.hlines(jnp.arange(1,8), -0.1, 8.1, color='w', lw=0.75)
        ax.minorticks_on()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator())
        ax.yaxis.set_minor_locator(ticker.MultipleLocator())
        ax.grid(which="both", linewidth=1.5, color="w")
        ax.tick_params(which="minor", length=0)
ax_cb = fig.add_subplot(gs[:-2, -2:])
ax_cb.set_title(r"Scale of $\log\theta$", y=0.6)
cax_cb = ax_cb.inset_axes((0.1, 0.45, 0.8, 0.1))
ax_cb.axis('off')
fig.colorbar(t, cax=cax_cb, orientation='horizontal')
ax_true = fig.add_subplot(gs[-2:, -2:])
ax_true.set_aspect('equal')
ax_true.matshow(ab.theta_true.log().reshape(8, 8), **plt_kwargs)
ax_true.minorticks_on()
ax_true.set_xticks([])
ax_true.set_yticks([])
ax_true.xaxis.set_minor_locator(ticker.MultipleLocator())
ax_true.yaxis.set_minor_locator(ticker.MultipleLocator())
ax_true.set_title(r"True $\theta$")
ax_true.grid(which="both", linewidth=1.5, color="w")
ax_true.tick_params(which="minor", length=0)
plt.show()

# %%
del pts
gc.collect()
from nak_torch.tools import kernel
def grad_log_p(pts: torch.Tensor):
    pts_grad = pts.clone()
    pts_grad.requires_grad_()
    fitness = log_p(pts_grad)   # shape (N,)
    grads, = torch.autograd.grad(fitness.sum(), pts_grad)
    return grads

# %%
stein_kernel_mat = kernel.stein_kernel_mat_factory(
    grad_log_p,
    kernel.sqexp_kernel_elem,
    is_grad_vectorized=True
)

# %%
stein_kernel_bandwidth = 5.0

idx = []
df_dict = {k: [] for k in [
    "kernel_mat",
    "log_p_evals",
    "KOQ_wts",
    "norm_KOQ_wts",
    "norm_simplex_wts",
    "stein_mat",
    "KSD_unif_wts",
    "KSD_KOQ_wts",
    "KSD_norm_KOQ_wts",
    "KSD_norm_simplex_wts",
]}

with torch.no_grad():
    pts = trajectories_svgd[-1]
    for pts, alg_name in tqdm([
        (trajectories_msip[-1], "MSIP"),
        (trajectories_svgd[-1], "SVGD")
    ]):
        kernel_mat = sqexp_kernel_matrix(pts, kernel_bandwidth**2)
        log_p_evals = log_p(pts)
        log_p_evals -= log_p_evals.max()
        wts = torch.linalg.lstsq(kernel_mat, log_p_evals.exp()).solution
        norm_wts = wts / wts.sum()
        simplex_wts = torch.linalg.lstsq(kernel_mat, torch.ones_like(wts)).solution
        simplex_wts /= simplex_wts.sum()
        stein_mat = stein_kernel_mat(pts, stein_kernel_bandwidth, None)

        ksd_unif = (stein_mat.sum() / (pts.shape[0]**2)).sqrt()
        ksd_koq = torch.sqrt((stein_mat @ wts) @ wts)
        ksd_norm_koq = torch.sqrt((stein_mat @ norm_wts) @ norm_wts)
        ksd_norm_proj = torch.sqrt((stein_mat @ simplex_wts) @ simplex_wts)
        idx.append(alg_name)
        df_dict["kernel_mat"].append(kernel_mat)
        df_dict["log_p_evals"].append(log_p_evals)
        df_dict["KOQ_wts"].append(wts)
        df_dict["norm_KOQ_wts"].append(norm_wts)
        df_dict["norm_simplex_wts"].append(simplex_wts)
        df_dict["stein_mat"].append(stein_mat)
        df_dict["KSD_unif_wts"].append(ksd_unif)
        df_dict["KSD_KOQ_wts"].append(ksd_koq)
        df_dict["KSD_norm_KOQ_wts"].append(ksd_norm_koq)
        df_dict["KSD_norm_simplex_wts"].append(ksd_norm_proj)

# %%
df_ksd = {k:[x.item() for x in v] for (k,v) in df_dict.items() if k.startswith("KSD")}
df = pd.DataFrame(df_ksd, index=idx)
df

# %%
