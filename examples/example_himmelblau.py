# %%
import torch
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from functions import himmelblau
from nak_torch.algorithms import msip, svgd
from nak_torch.algorithms.msip import MSIPFredholm, MSIPQuadGradientFree
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
from datetime import datetime
from nak_torch.tools.kernel import kernel_optimal_weight_factory, default_kernel_matrix

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
    "kernel_length_scale": 0.5,
    "init_particles": init_particles,
    "n_particles": n_particles,
    "dim": 2,
    "lr": 0.6,
    "kernel_diag_infl": 1e-8,
}

# %%
estimator_fredholm = MSIPFredholm(
    gradient_decay=0.85,
    log_dens_grad_val=torch.vmap(torch.func.grad_and_value(log_density))
)

trajectories_fr, trajectories_wts_fr = msip(
    estimator_fredholm,
    # n_particles=n_particles,
    # init_particles=init_particles,
    n_steps=100,          # now interpreted as "epochs" (passes over all particles)
    # lr=0.6,
    # noise=0.05,          # currently unused, kept for compatibility
    # kernel_length_scale=0.5,
    # inner_tol=1e-4,      # equilibrium tolerance for a particle
    # max_inner_steps=1000,  # max inner iterations per particle
    # kernel_diag_infl=1e-8,
    # seed=,
    **params
)

# %%
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
    # n_particles=25,
    # init_particles=init_particles,
    n_steps=100,          # now interpreted as "epochs" (passes over all particles)
    # dim=2,
    # bounds=(-20, 20),
    # lr=0.6,
    # noise=0.05,          # currently unused, kept for compatibility
    # kernel_length_scale=0.5,
    # inner_tol=1e-4,      # equilibrium tolerance for a particle
    # max_inner_steps=1000,  # max inner iterations per particle
    # kernel_diag_infl=1e-8,
    # seed=,
    **params
)

# %%
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
    lambda b: spherical_MC_radial_Laguerre(b, 5, 2, 2)
)
n_particles = 25
trajectories_gf,w = msip(
    estimator,
    # n_particles=n_particles,
    # init_particles=init_particles,
    n_steps=50,          # now interpreted as "epochs" (passes over all particles)
    # dim=2,
    # bounds=(-20, 20),
    # lr=0.6,
    # kernel_length_scale=0.5,
    # kernel_diag_infl=1e-8,
    seed=1,
    **params
)

pts_gf = trajectories_gf[-1]
wts_gf = kernel_optimal_weight_factory(pts_gf, log_density(pts_gf), default_kernel_matrix(pts_gf, params["kernel_length_scale"]))
plt.contourf(X,Y,Z, levels=20, cmap="Grays")
plt.scatter(pts_gf[:,0], pts_gf[:,1], c=wts_gf)

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
    # fpath = f"results/gif/gradfree_{function_name}_particles_{now_stamp}.mp4"
    # animate_trajectories_box(
    #     log_density, trajectories_gf, 1, bounds,
    #     save_path=fpath, writer="ffmpeg"
    # )


# %%
