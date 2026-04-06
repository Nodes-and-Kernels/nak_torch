# %%

import numpy as np
from viz_tools import animate_trajectories_box
from functions import gaussian_kde
from nak_torch.algorithms import msip_greedy
from datetime import datetime

save_gif = True
now_stamp = datetime.now()
algorithm_name = "msip_greedy" # Other choices: msip, msip_gs, msip_ni
function_name = "sqexp_kde" # Other choices: mixture_of_gaussians, himmelblau

sigma = 0.3
npz_file = np.load('datasets/two_bananas.npz')
dataset = 10*npz_file['X'].T - np.mean(10*npz_file['X'].T)
log_density = gaussian_kde(sigma,dataset)
bounds = (-4.,4.)



trajectories = msip_greedy(
    log_density,
    n_particles=25,
    n_steps=40, # "epochs" (passes over all particles)
    dim=2,
    bounds=bounds,   # [a,b]^d
    gradient_informed = True,
    lr=0.8,
    noise=0.05,          # currently unused
    kernel_bandwidth=0.3,
    bandwidth_factor = 0.001,
    inner_tol=1e-6,      # equilibrium tolerance for a particle
    max_inner_steps=5,  # max inner iterations per particle
    seed=0,
    device="cpu"
)



if save_gif:
    fpath = f"results/gif/{algorithm_name}_{function_name}_particles_{now_stamp}.gif"
    animate_trajectories_box(
        log_density, trajectories, 50, bounds,
        save_path=fpath
    )
# %%
