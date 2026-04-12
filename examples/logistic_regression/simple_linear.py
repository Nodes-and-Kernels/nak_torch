# %%
import torch
from nak_torch.algorithms import msip, msip_gs, svgd
import matplotlib.pyplot as plt
from nak_torch import LogisticRegressionModel
from nak_torch.tools import pyro_tools
from pyro.infer import mcmc

from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
)
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre

import os

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

# %%
data_path = os.path.join(os.path.dirname(__file__), "data", "simple_linear.npy")
regression_model = LogisticRegressionModel(data_path, None, hyperprior_b=0.01)
log_dens = regression_model.to_log_dens(use_compiled=False)

plt.scatter(regression_model.data[1], regression_model.data[2], c=regression_model.labels, alpha=0.4)
plt.show()

# %%
n_particles, state_dim = 20, regression_model.dim
coeff_init = torch.randn((n_particles, regression_model.dim - 1))
alpha_init = torch.log(regression_model.hyperprior.sample((n_particles,)))
init_particles = torch.column_stack((coeff_init, alpha_init))
log_dens(init_particles)  # test eval

# %%
kernel_length_scale = 0.05
bounds = (-100.0, 100.0)
gradient_decay = 0.95
lr_msip = 80e-2
kernel_diag_infl = 1e-7
n_steps = 1000
grad_val_log_p = torch.vmap(torch.func.grad_and_value(log_dens))

@torch.compile(dynamic=False)
def mc_quad_rule(batch_size: int, N_quad: int = 500, dim: int = 4):
    pts = torch.randn((batch_size, N_quad, dim), dtype=torch.get_default_dtype())
    wts = torch.ones((batch_size, N_quad), dtype=torch.get_default_dtype()).div_(N_quad)
    return pts, wts

@torch.compile(dynamic=False)
def spherical_quad(batch_size: int, N_spherical: int = 10, N_radial: int = 3, dim: int = 4):
    pts, wts = spherical_MC_radial_Laguerre(batch_size, N_spherical, dim, N_radial)
    return pts, wts


msip_f = MSIPFredholm(gradient_decay, grad_val_log_p)
msip_gi = MSIPQuadGradientInformed(grad_val_log_p, mc_quad_rule, gradient_decay)

# %%
trajectories_msip, traj_wts_msip = msip(
    msip_f,
    n_particles,
    n_steps,
    dim=state_dim,
    lr=lr_msip,
    init_particles=init_particles[:n_particles],
    kernel_length_scale=kernel_length_scale,
    is_log_density_batched=True,
    kernel_diag_infl=kernel_diag_infl,
    bounds=bounds,
    keep_all=True,
    compile_step=False,
    verbose=True,
)

# %%
msip_idx = 999
msip_final_pts, msip_final_wts = trajectories_msip[msip_idx], traj_wts_msip[msip_idx]
logit_out = msip_final_pts[:,:-1] @ regression_model.data
prob_out = torch.nn.functional.sigmoid(logit_out)

fig, axs = plt.subplots(4,5,figsize=(5*1.25,4.5*1.25))
sc_data = regression_model.data[1:]
for i in range(4):
    for j in range(5):
        ax = axs[i,j]
        idx = (5*i + j)
        prob_ij = prob_out[idx]
        wt = msip_final_wts[idx]
        ax.scatter(sc_data[0], sc_data[1], c=prob_ij, alpha=0.1)
        ax.set_axis_off()
        ax.set_title("{:.2e}".format(wt), fontdict={'fontsize': 10})
fig.suptitle("Different regression outcomes, MSIP wt as title")
plt.show()

# %%
plt.scatter(sc_data[0], sc_data[1], c=regression_model.labels)

# %%
n_steps_hmc = 1000
pyro_model = pyro_tools.pyro_model_factory(regression_model, 4)
pyro_data = torch.concat((regression_model.data, regression_model.labels.reshape(1,-1)))
hmc_kernel = mcmc.NUTS(pyro_model)
mcmc_setup = mcmc.MCMC(hmc_kernel, num_samples=n_steps_hmc, warmup_steps=100)
mcmc_setup.run(pyro_data)

# %%
hmc_samples = mcmc_setup.get_samples()["theta"]
thin_samples = hmc_samples[::(len(hmc_samples) // 20)]
logit_out = thin_samples @ regression_model.data
prob_out = torch.nn.functional.sigmoid(logit_out)

fig, axs = plt.subplots(4,5,figsize=(5*1.25,4.5*1.25))
sc_data = regression_model.data[1:]
for i in range(4):
    for j in range(5):
        ax = axs[i,j]
        idx = (5*i + j)
        prob_ij = prob_out[idx]
        ax.scatter(sc_data[0], sc_data[1], c=prob_ij, alpha=0.1)
        ax.set_axis_off()
        # ax.set_title("{:.2e}".format(wt), fontdict={'fontsize': 10})
fig.suptitle("Different regression outcomes, MSIP wt as title")
plt.show()
