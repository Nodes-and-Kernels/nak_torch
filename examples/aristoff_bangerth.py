# %%
import math
import torch
from functions import aristoff_bangerth as ab, build_aristoff_bangerth
import nak_torch
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm
from matplotlib import ticker
import gc
import matplotlib.pyplot as plt
from nak_torch.tools.kernel import sqexp_kernel_matrix
from tqdm import tqdm
import pandas as pd
from nak_torch.tools.types import BatchGradLogDensityEvaluator
from nak_torch.tools import pyro_tools
from pyro.infer import mcmc

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

# %%
def plot_samples(pts, max_side_len = 6):
    n_particles = pts.shape[0]
    side_len = min(max_side_len, int(math.floor(math.sqrt(n_particles))))
    pts = pts[:side_len**2]
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
    fig.colorbar(t, cax=cax_cb, orientation='horizontal') # type: ignore
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
    return fig


# %%
use_compiled = True
model = build_aristoff_bangerth(use_compiled=use_compiled, dtype=torch.float64)
log_p = model.to_log_dens(use_compiled=use_compiled)
log_th = torch.randn(25, 64, dtype=torch.float64)
test_out = log_p(log_th, None)

# %%
def _tmp_log_p(log_theta, arg: None):
    ret = log_p(log_theta, arg)
    return ret.sum(), ret

grad_log_p = torch.func.grad(lambda t,a: log_p(t, a).sum())
grad_val_log_p = torch.func.grad(_tmp_log_p, has_aux=True)
test_grad = grad_log_p(log_th, None)
test_grad_2, test_out_2 = grad_val_log_p(log_th, None)

# %%
del log_th, test_out, test_grad, test_grad_2, test_out_2
gc.collect()

# %%
n_particles, n_steps, dim = 25, 25, 64
kernel_bandwidth = 0.75

torch.manual_seed(1)
init_particles = 2 * torch.randn(
    (n_particles, dim),
    dtype=torch.float64,
)  # Sample from prior

default_kwargs = {
    "dim": dim,
    "bounds": (-8, 8),
    "n_steps": n_steps,
    "n_particles": n_particles,
    "keep_all": False,
    "lr": 1e-1,
    "kernel_lengthscale": 0.1,
    "init_particles": init_particles,
    "gradient_decay": 0.95,
    "kernel_diag_infl": 1e-6,
}

# %%
msip_kwargs = default_kwargs.copy()
msip_kwargs["lr"] = 1e-2
msip_kwargs["kernel_lengthscale_quantile"] = 0.05
msip = MSIP(**msip_kwargs)
target_msip_fr = MSIPFredholm(log_dens_grad_val=grad_val_log_p, **msip_kwargs)

# %%
trajectories_pts_msip_fr, trajectories_wts_msip_fr = nak_torch.nak(target_msip_fr, msip, **msip_kwargs)

# %%
n_steps_hmc = 100
pyro_model = pyro_tools.PyroModel(model, dim)
hmc_kernel = mcmc.NUTS(pyro_model)
mcmc_setup = mcmc.MCMC(hmc_kernel, num_samples=n_steps_hmc, warmup_steps=10)
mcmc_setup.run(model.true_obs)

hmc_samples = mcmc_setup.get_samples()["theta"]

# %%
target_svgd = BatchGradLogDensityEvaluator(
    log_p, is_grad=False, is_batched=True
)

svgd = SVGD(
    kernel_lengthscale_quantile=0.5,  # Median heuristic
    **msip_kwargs
)
svgd_kwargs = msip_kwargs.copy()
svgd_kwargs["lr"] = 1e-1
svgd_kwargs["n_steps"] = 100

trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    **svgd_kwargs
)

# %%
pts_msip = trajectories_pts_msip_fr[-1] - init_particles
fig = plot_samples(pts_msip)
fig.suptitle("MSIP Samples")
plt.show()

# %%
pts_hmc = hmc_samples[10::3]
fig = plot_samples(pts_hmc)
fig.suptitle("HMC Samples")
plt.show()

# %%
pts_svgd = trajectories_pts_svgd[-1]
fig = plot_samples(pts_svgd)
fig.suptitle("SVGD Samples")
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
