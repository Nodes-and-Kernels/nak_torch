# %%
import os
from urllib.request import urlretrieve
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
import scipy.io
import numpy as np

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)

# %%
DATA_URL = "https://raw.githubusercontent.com/DartML/Stein-Variational-Gradient-Descent/refs/heads/master/data/covertype.mat"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "covertype.npy")

def download_file(data_url: str = DATA_URL, data_path: str = DATA_PATH):
    urlretrieve(data_url, data_path)
    data_mat = scipy.io.loadmat(data_path)
    data_arr = data_mat['covtype']
    # Flip first col to be (0,1) instead of (2,1) (where 2 is false)
    covariates = data_arr[:, 1:]
    labels = -1 * (data_arr[:, 0] - 2)
    data_arr = np.column_stack((covariates, labels))
    # Save
    np.save(data_path, data_arr)

if not os.path.isfile(DATA_PATH):
    download_file()

# %%
data_path = DATA_PATH
regression_model = LogisticRegressionModel(data_path, None, hyperprior_b=0.01, train_proportion=0.8, sum_bernoulli=True)
log_dens = regression_model.to_log_dens(use_compiled=False)

# %%
N_plot = 10000
plt.scatter(regression_model.train_data[2,:N_plot], regression_model.train_data[3, :N_plot], c=regression_model.train_labels[:N_plot], alpha=0.2)
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
gradient_decay = 0.75
lr_msip = 1e-1
kernel_diag_infl = 1e-6
n_steps = 20
grad_val_log_p = torch.vmap(torch.func.grad_and_value(log_dens))

@torch.compile(dynamic=False)
def mc_quad_rule(batch_size: int, N_quad: int = 500, dim: int = 56):
    pts = torch.randn((batch_size, N_quad, dim), dtype=torch.get_default_dtype())
    wts = torch.ones((batch_size, N_quad), dtype=torch.get_default_dtype()).div_(N_quad)
    return pts, wts

@torch.compile(dynamic=False)
def spherical_quad(batch_size: int, N_spherical: int = 10, N_radial: int = 3, dim: int = 56):
    pts, wts = spherical_MC_radial_Laguerre(batch_size, N_spherical, dim, N_radial)
    return pts, wts


msip_f = MSIPFredholm(gradient_decay, grad_val_log_p)
msip_gi = MSIPQuadGradientInformed(grad_val_log_p, mc_quad_rule, gradient_decay)

# %%
trajectories_msip, traj_wts_msip = msip(
    msip_gi,
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
    compile_step=True,
    verbose=True,
)
trajectories_msip[-1]

# %%
msip_end = trajectories_msip[-1]
dist_end = torch.sqrt(torch.sum(torch.square_(msip_end[None,:] - msip_end[:,None]), -1))
lower_tri_idx = torch.tril_indices(*dist_end.shape, -1)
lower_tri_dist = dist_end[*lower_tri_idx]
plt.hist(lower_tri_dist)

# %%
bce_logit_v = torch.vmap(torch.nn.functional.binary_cross_entropy_with_logits, in_dims=(0,None))

# @torch.compile
def bce_logit_t(traj_t):
    logits_t = traj_t[:,:-1] @ regression_model.test_data
    return bce_logit_v(logits_t, regression_model.test_labels)
# bce_logit_traj = torch.vmap(bce_logit_t)
bce_traj = torch.stack([bce_logit_t(trajectories_msip[j]) for j in range(trajectories_msip.shape[0])])
# logits_t = trajectories_msip[:,:,:-1].reshape(-1, trajectories_msip.shape[-1] - 1) @ regression_model.data
# bce_traj = bce_logit_v(logits_t, regression_model.labels).reshape(*trajectories_msip.shape[:2], -1)
# print("BCE t=0: {}, BCE t=T: {}".format(bce_0.mean(), bce_T.mean()))

fig, ax = plt.subplots()
for particle_idx in range(n_particles):
    ax.plot(bce_traj[:,particle_idx], alpha= 0.4)
plt.show()

# %%
def accuracy(coeffs):
    data, labels = regression_model.test_data, regression_model.test_labels
    prob = torch.sigmoid(coeffs[:-1] @ data)
    pred_labels = prob > 0.5
    print(pred_labels.sum())
    N_true = torch.sum(pred_labels == labels)
    return N_true / data.shape[1]

accuracy_v = torch.vmap(accuracy)
accuracy_v(trajectories_msip[-1])

# %%

trajectories_msip, traj_wts_msip = svgd(
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
    compile_step=True,
    verbose=True,
)
