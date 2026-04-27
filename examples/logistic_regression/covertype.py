# %%
import os
from urllib.request import urlretrieve
import torch
import nak_torch
from nak_torch.algorithms import MSIP, MSIPGS, SVGD
import matplotlib.pyplot as plt
from nak_torch import LogisticRegressionModel
from nak_torch.tools import pyro_tools
from pyro.infer import mcmc
from tqdm import tqdm

from nak_torch.algorithms.msip import (
    MSIPFredholm,
    MSIPQuadGradientInformed,
    MSIPQuadGradientFree,
)

from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
import scipy.io
import numpy as np

from nak_torch.tools.types import BatchGradLogDensityEvaluator

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
    data_arr = data_mat["covtype"]
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
regression_model = LogisticRegressionModel(
    data_path, None, hyperprior_b=0.01, train_proportion=0.8, reduction="sum"
)
log_dens = regression_model.to_log_dens(use_compiled=True)

# %%
N_plot = 10000
plt.scatter(
    regression_model.train_data[:N_plot, 2],
    regression_model.train_data[:N_plot, 3],
    c=regression_model.train_labels[:N_plot],
    alpha=0.2,
)
plt.show()

# %%
N_PARTICLES = 100
STATE_DIM = regression_model.dim
alpha_init = regression_model.hyperprior.sample((N_PARTICLES, 1))
log_alpha_init = alpha_init.log()
coeff_init = torch.randn((N_PARTICLES, STATE_DIM - 1)) / alpha_init.sqrt()
init_particles = torch.column_stack((coeff_init, log_alpha_init))
log_dens(init_particles)  # test eval

# %%
BATCH_SIZE = 50
train_data_loader = regression_model.get_data_loader(False, batch_size=BATCH_SIZE)

# %%
grad_val_log_p = torch.vmap(torch.func.grad_and_value(log_dens), in_dims=(0, None))


@torch.compile(dynamic=False)
def mc_quad_rule(batch_size: int, N_quad: int = 500, dim: int = 56):
    pts = torch.randn((batch_size, N_quad, dim), dtype=torch.get_default_dtype())
    wts = torch.ones((batch_size, N_quad), dtype=torch.get_default_dtype()).div_(N_quad)
    return pts, wts


@torch.compile(dynamic=False)
def spherical_quad(
    batch_size: int, N_spherical: int = 10, N_radial: int = 3, dim: int = 56
):
    pts, wts = spherical_MC_radial_Laguerre(batch_size, N_spherical, dim, N_radial)
    return pts, wts


# %%
KERNEL_LENGTHSCALE = 0.1
GRADIENT_DECAY = 0.9
KERNEL_DIAG_INFL = 1e-5
KERNEL_QUANTILE = 0.01

msip = MSIP(
    dim=STATE_DIM,
    n_particles=N_PARTICLES,
    kernel_diag_infl=KERNEL_DIAG_INFL,
    kernel_lengthscale=KERNEL_LENGTHSCALE,
    kernel_lengthscale_quantile=KERNEL_QUANTILE,
)

target_msip_f = MSIPFredholm(GRADIENT_DECAY, grad_val_log_p)
target_msip_gi = MSIPQuadGradientInformed(grad_val_log_p, mc_quad_rule, GRADIENT_DECAY)

# %%
BOUNDS = (-100.0, 100.0)
N_STEPS = 6000
LR_MSIP = 0.01
# trajectories_pts_msip_fr, trajectories_wts_msip_fr = nak_torch.nak(
#     target_msip_f,
#     msip,
#     n_steps=N_STEPS,
#     lr=LR_MSIP,
#     init_particles=init_particles,
#     get_target_args=iter(train_data_loader),
#     bounds=BOUNDS,
# )

# %%
msip_end = trajectories_pts_msip_fr[-1]
dist_end = torch.sqrt(
    torch.sum(torch.square_(msip_end[None, :] - msip_end[:, None]), -1)
)
lower_tri_idx = torch.tril_indices(*dist_end.shape, -1)
lower_tri_dist = dist_end[*lower_tri_idx]
plt.hist(lower_tri_dist)

# %%
bce_logit_v = torch.vmap(
    torch.nn.functional.binary_cross_entropy_with_logits, in_dims=(0, None)
)


# @torch.compile
def bce_logit_t(traj_t):
    logits_t = traj_t[:, :-1] @ regression_model.test_data.T
    return bce_logit_v(logits_t, regression_model.test_labels)


bce_logit_traj = torch.vmap(bce_logit_t)
bse_traj_list = []
for j in tqdm(range(trajectories_pts_msip_fr.shape[0])):
    bse_traj_list.append(bce_logit_t(trajectories_pts_msip_fr[j]))
bce_traj = torch.stack(bse_traj_list)
# logits_t = trajectories_msip[:,:,:-1].reshape(-1, trajectories_msip.shape[-1] - 1) @ regression_model.data
# bce_traj = bce_logit_v(logits_t, regression_model.labels).reshape(*trajectories_msip.shape[:2], -1)
# print("BCE t=0: {}, BCE t=T: {}".format(bce_0.mean(), bce_T.mean()))

# %%
fig, ax = plt.subplots()
for particle_idx in range(N_PARTICLES):
    ax.loglog(bce_traj[:, particle_idx], alpha=0.4)
plt.show()


# %%
def accuracy(coeffs):
    data, labels = regression_model.test_data, regression_model.test_labels
    prob = torch.sigmoid(coeffs[:-1] @ data.T)
    pred_labels = prob > 0.5
    return torch.mean((pred_labels == labels).to(torch.float64))

accuracy_v = torch.vmap(accuracy)
# accuracy_v(trajectories_pts_msip_fr[-1])

# %%
svgd = SVGD(
    STATE_DIM,
    N_PARTICLES,
    kernel_lengthscale_quantile=0.5
)

target_svgd = BatchGradLogDensityEvaluator(
    log_dens, is_grad=False, is_batched=True
)

# %%
LR_SVGD = 0.05
trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    n_steps=N_STEPS,
    lr=LR_SVGD,
    init_particles=init_particles,
    get_target_args=iter(train_data_loader),
    # bounds=BOUNDS,
)

# %%
accuracy(trajectories_pts_svgd[-1].mean(0))

# %%
