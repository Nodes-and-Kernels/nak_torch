# %%
import torch
import nak_torch
from nak_torch.algorithms import MSIP, SVGD
from nak_torch.algorithms.msip import MSIPFredholm
import bnn_impl as bnn
from nak_torch.tools.types import BatchLogDensityGradEvaluator
from nak_torch.tools.util import infinite_iter

# %%
# Data loading
DATASET = "two_bananas"
PROP_TRAIN = 0.8
data_train, labels_train, data_test, labels_test = bnn.load_dataset(
    DATASET, train_ratio=PROP_TRAIN, seed=0
)

# %%
N_LAYERS = 1
HIDDEN_DIM = 50
BETA = 1.0
LAMBDA = 0.2
BATCH_SIZE = 64

bnn_posterior = bnn.BNNClassifierPosterior(
    data=data_train,
    labels=labels_train,
    beta=BETA,
    hidden_dim=HIDDEN_DIM,
    weight_decay=LAMBDA**2,
)

data_loader = bnn_posterior.get_data_loader(BATCH_SIZE, shuffle = True)

# %%
N_PARTICLES = 100
init_particles = torch.randn(N_PARTICLES, bnn_posterior.dimension).double() / LAMBDA

# %%
b_post_ev = bnn_posterior(init_particles, None)

# %%
def tmp_post(x, a = None):
    ret = bnn_posterior(x, a)
    return ret.sum(), ret
log_dens_grad_val = torch.func.grad(tmp_post, has_aux=True)

# %%
GRADIENT_DECAY = 1.0
KERNEL_LENGTHSCALE = 0.1
target_msip_fr = MSIPFredholm(GRADIENT_DECAY, log_dens_grad_val)

msip = MSIP(
    dim = bnn_posterior.dimension,
    n_particles= N_PARTICLES,
    kernel_lengthscale = KERNEL_LENGTHSCALE,
)


# %%
BOUNDS = (-1000., 1000.)
N_STEPS = 10000
LR_MSIP = 1e-3
trajectories_pts_msip_fr, trajectories_wts_msip_fr = nak_torch.nak(
    target_msip_fr,
    msip,
    n_steps=N_STEPS,
    lr=LR_MSIP,
    init_particles=init_particles,
    get_target_args=infinite_iter(data_loader),
    bounds=BOUNDS,
    keep_all=True
)

# %%
bnn.eval_function_trajectories(bnn_posterior, trajectories_pts_msip_fr.double(), "MSIP")

# %%
target_svgd = BatchLogDensityGradEvaluator(
    bnn_posterior, is_grad=False, is_batched=True
)

svgd = SVGD(
    dim = bnn_posterior.dimension,
    n_particles= N_PARTICLES,
    kernel_lengthscale_quantile= 0.5
)


# %%
trajectories_pts_svgd = nak_torch.nak(
    target_svgd,
    svgd,
    n_steps=N_STEPS,
    lr=LR_MSIP,
    init_particles=init_particles,
    get_target_args=infinite_iter(data_loader),
    bounds=BOUNDS,
    keep_all=True
)

# %%
bnn.eval_function_trajectories(bnn_posterior, trajectories_pts_svgd.double(), "SVGD") # type: ignore

# %%
trajectories_dict = {"MSIP": trajectories_pts_msip_fr, "SVGD": trajectories_pts_svgd}
objective_fns_dict = {"MSIP": bnn_posterior, "SVGD": bnn_posterior}

# Visualization
bnn.plot_boundaries(trajectories_dict, objective_fns_dict, bnn_posterior.data, bnn_posterior.labels)

bnn.evaluate(trajectories_dict, objective_fns_dict, data_test, labels_test, "test")

# %%
# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Config
    DATASET = "two_bananas"
    MODEL_CLASS = "bnn"
    HIDDEN_DIM = 50
    N_LAYERS = 1
    N_TRAIN = 0.8  # train-test split ratio
    N_PARTICLES = 250
    N_STEPS = 1000
    BETA = 1.0  # beta in x-> exp(-beta^{-1}V(x))
    LAMBDA2 = 0.00005  # lambda in prior;0005
    # lambda close to 0 means weak prior
    LR_SVGD = 100e-2
    LR_MSIP = 100e-2
    SIGMA = 0.5

    # Data loading
    X_train, Y_train, X_test, Y_test = load_dataset(
        DATASET, train_ratio=N_TRAIN, seed=0
    )

    # Objective loading
    obj_msip = make_objective(
        X_train,
        Y_train,
        model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        beta=BETA,
        lambda2=LAMBDA2,
    )
    obj_svgd = make_objective(
        X_train,
        Y_train,
        model_class=MODEL_CLASS,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        beta=BETA,
        lambda2=LAMBDA2,
    )

    dimension = obj_msip.total_numel

    # Shared inititializtion
    init_particles = 5 * torch.randn(N_PARTICLES, dimension).double()

    # Run MSIP
    post_log_dens_grad_val = torch.func.grad_and_value(obj_msip)
    msip_fredholm = MSIPFredholm(1.0, post_log_dens_grad_val)

    trajectories_msip, wts_msip = msip(
        obj_msip,
        N_PARTICLES,
        N_STEPS,
        dim=dimension,
        lr=LR_MSIP,
        init_particles=init_particles,
        kernel_length_scale=SIGMA,
        is_log_density_batched=True,
        kernel_diag_infl=1e-8,
        bounds=(-100.0, 100.0),
        gradient_decay=1.0,
        keep_all=True,
        compile_step=False,
        verbose=True,
    )

    # Run SVGD

    trajectories_svgd = svgd(
        obj_svgd,
        N_PARTICLES,
        N_STEPS,
        dimension,
        LR_SVGD,
        seed=None,
        device=None,
        init_particles=init_particles,
        kernel_length_scale=SIGMA,
        keep_all=True,
        is_log_density_batched=True,
        verbose=True,
    )

    trajectories_dict = {"MSIP": trajectories_msip, "SVGD": trajectories_svgd}
    objective_fns_dict = {"MSIP": obj_msip, "SVGD": obj_svgd}

    # Optimization diagnostics
    eval_function_trajectories(obj_msip, trajectories_msip, "MSIP")
    eval_function_trajectories(obj_svgd, trajectories_svgd, "SVGD")

    # Visualization
    plot_boundaries(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_mean_prediction(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_diversity_curve(trajectories_dict)

    # Evaluation on a dataset
    evaluate(trajectories_dict, objective_fns_dict, X_train, Y_train, "train")
    evaluate(trajectories_dict, objective_fns_dict, X_test, Y_test, "test")
