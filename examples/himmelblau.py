# %%
import torch
import matplotlib.pyplot as plt
import nak_torch
from viz_tools import animate_trajectories_box
from functions import himmelblau
from nak_torch.algorithms import msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
from datetime import datetime
from nak_torch.tools.kernel import kernel_optimal_weight_factory, DEFAULT_KERNEL_MATRIX

save_gif = False
algorithm_name = "msip_ni"
function_name = "himmelblau"
log_density = himmelblau(50.0)

n_particles = 25

# %%
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)
torch.manual_seed(19230182)
init_particles = torch.randn((n_particles, 2)) + 8.0
params = {
    "bounds": (-15, 15),
    "kernel_length_scale": 0.2,
    "init_particles": init_particles,
    "n_particles": n_particles,
    "dim": 2,
    "lr": 0.8,
    "kernel_diag_infl": 1e-5,
}

# %%
estimator_fredholm = MSIPFredholm(
    gradient_decay=0.95,
    log_dens_grad_val=torch.vmap(torch.func.grad_and_value(log_density))
)

trajectories_fr, trajectories_wts_fr = msip(
    estimator_fredholm,
    n_steps=200,
    use_quantile_length_scale=0.0,
    **params
)

Ngrid = 1000
x = y = torch.linspace(-5, 5, Ngrid)
X,Y = torch.meshgrid(x,y,indexing="ij")
XY = torch.stack((X.flatten(), Y.flatten()), dim=1)
Z = log_density(XY).reshape(Ngrid,Ngrid).exp()

pts_fr = trajectories_fr[-1]
wts_fr = trajectories_wts_fr[-1]
plt.contourf(X,Y,Z, levels=20, cmap="Grays")
s = plt.scatter(
    pts_fr[:,0],
    pts_fr[:,1],
    c=wts_fr.log(), edgecolors="k", linewidths=0.5
)

plt.show()


# %%
trajectories_svgd = svgd(
    log_density,
    n_steps=100,
    use_quantile_length_scale=0.5,
    **params
)

plt.contourf(X,Y,Z, levels=20, cmap="Grays")
pts_svgd = trajectories_svgd[-1]
s = plt.scatter(
    pts_svgd[:,0],
    pts_svgd[:,1],
    edgecolors="k", linewidths=0.5
)

plt.show()


# %%
estimator = MSIPQuadGradientFree(
    log_density,
    lambda b: spherical_MC_radial_Laguerre(b, N_spherical=5, d=2, N_radial=2)
)
params_gf = params.copy()
params_gf['lr'] = 0.5
n_particles = 25
trajectories_gf,w = msip(
    estimator,
    n_steps=500,
    seed=1,
    use_quantile_length_scale=0.0,
    **params_gf
)

pts_gf = trajectories_gf[-1]
wts_gf = kernel_optimal_weight_factory(pts_gf, log_density(pts_gf), DEFAULT_KERNEL_MATRIX(pts_gf, params["kernel_length_scale"]))
plt.contourf(X,Y,Z, levels=20, cmap="Grays")
plt.scatter(pts_gf[:,0], pts_gf[:,1], c=wts_gf)
plt.show()

# %%
batch_log_dens = torch.vmap(log_density)
batch_grad_log_dens = torch.vmap(torch.func.grad(log_density))
def ksd_kernel_elem(x: torch.Tensor, y: torch.Tensor, sigma: float):
    return torch.reciprocal(1 + (x - y).div(sigma).square().sum())

ksd_lengthscale = 0.25

metrics = {
    "KSD": nak_torch.metrics.KernelSteinDiscrepancy(batch_grad_log_dens, ksd_lengthscale, kernel_elem=ksd_kernel_elem),
    "rESS": nak_torch.metrics.RelativeESS(batch_log_dens),
    "CrossEntropy": nak_torch.metrics.CrossEntropy(batch_log_dens)
}
for metric_name, metric in metrics.items():
    print(
        "\n\n" + metric_name,
        "\nFredholm, weighted",
        metric(pts_fr, wts_fr).item(),
        "Fredholm, unweighted",
        metric(pts_fr).item(),
        "\nSVGD",
        metric(pts_svgd).item(),
        "\nGradient-free, weighted",
        metric(pts_gf, wts_gf).item(),
        "\nGradient-free, unweighted",
        metric(pts_gf).item(),
    )

# %%
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'NewComputerModern10'
fig, axs = plt.subplots(1, 4, figsize=(10.5,2.5))
pt_list = [init_particles, pts_svgd, pts_fr, pts_gf]
wt_list = [None, None, wts_fr, wts_gf]
extrema_pts = [(pt.min(), pt.max()) for pt in pt_list]
g_min, g_max = [m(x[i] for x in extrema_pts) for (m,i) in [(min,0), (max,1)]]
extrema_wts = [(w.min(), w.max()) for w in wt_list if w is not None]
vmin, vmax = [m(x[i] for x in extrema_wts) for (m,i) in [(min,0), (max,1)]]
titles = ["Initialization", "SVGD", "MSIP-1", "MSIP-GF"]
title_weights = [None, None, 'heavy', 'heavy']
for (ax, title, pt, wt, title_wt) in zip(axs, titles, pt_list, wt_list, title_weights):
    ax.set_axis_off()
    ax.set_xlim(g_min, g_max)
    ax.set_ylim(g_min, 1.05*g_max)
    ax.set_title(title, fontweight=title_wt, size=20)
    ax.contourf(X[:,40:],Y[:,40:],Z[:,40:], levels=20, cmap="Grays")
    s = 25 * (1. if wt is None else ((wt.abs()/wt.abs().max())).sqrt())
    c = torch.ones(pt.shape[0])/pt.shape[0] if wt is None else wt/wt.sum()
    ax.scatter(pt[:,0], pt[:,1], s=s, c=c, vmin=vmin, vmax=vmax)
fig.tight_layout()
fig.savefig("figs/himmelblau_ex.pdf", transparent=True)
plt.show()

# %%
# torch.set_default_device("cpu")
if save_gif:
    now_stamp = datetime.now()
    bounds = (-15.,40.)
    fpath = f"results/gif/fredholm_{function_name}_particles_{now_stamp}.mp4"
    animate_trajectories_box(
        log_density, trajectories_fr, 1, bounds,
        save_path=fpath, writer="ffmpeg"
    )

# %%
