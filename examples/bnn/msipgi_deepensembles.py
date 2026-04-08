import torch
from nak_torch.algorithms import msip_gs, deepensembles
from nak_torch.algorithms.msip import MSIPQuadGradientInformed
from nak_torch.tools.quadrature import spherical_MC_radial_Laguerre
from bnn_impl import (
    make_objective,
    eval_function_trajectories,
    plot_boundaries,
    plot_mean_prediction,
    plot_diversity_curve,
    evaluate,
    load_dataset,
)


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
    N_STEPS = 50
    BETA = 1.0  # beta in x-> exp(-beta^{-1}V(x))
    LAMBDA2 = 5e-6  # lambda in prior;0005
    # lambda close to 0 means weak prior
    GRADIENT_DECAY = 0.95
    LR_DE = 100e-2
    LR_MSIP = 100e-2
    SIGMA = 0.55

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
    obj_de = make_objective(
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
    init_particles = torch.randn(N_PARTICLES, dimension).double()

    # Run MSIP
    post_log_dens_grad_val = torch.func.vmap(torch.func.grad_and_value(obj_msip))

    @torch.compile(dynamic=False)
    def mc_quad_rule(batch_size: int, N_quad: int = 10, dim: int = dimension):
        # dim = dimension
        pts = torch.randn((batch_size, N_quad, dim), dtype=torch.float64)
        wts = torch.ones((batch_size, N_quad), dtype=torch.float64).div_(N_quad)
        return pts, wts

    @torch.compile(dynamic=False)
    def spherical_quad(
        batch_size: int,
        dimension: int = dimension,
        N_spherical: int = 5,
        N_radial: int = 3,
    ):
        pts, wts = spherical_MC_radial_Laguerre(
            batch_size, N_spherical, dimension, N_radial, dtype=torch.float64
        )
        return pts, wts

    # %%
    # kernel_length_scale = 1e-3
    # gradient_decay = 1.
    # mc_quad_rule
    msip_quadgrad = MSIPQuadGradientInformed(
        post_log_dens_grad_val, spherical_quad, gradient_decay=GRADIENT_DECAY
    )

    # trajectories_msip_qg, traj_wts_msip_qg = msip(
    #     msip_quadgrad, n_particles, n_steps, dim=2,
    #     lr=10., init_particles=init_particles[:n_particles],
    #     kernel_length_scale=kernel_length_scale,
    #     # is_log_density_batched=True,
    #     kernel_diag_infl=1e-8,
    #     bounds=(-1000, 1000),
    #     # gradient_decay=gradient_decay,
    #     keep_all=False,
    #     compile_step=False,
    #     verbose=True
    # )
    # msip_fredholm          = MSIPFredholm(1.0, post_log_dens_grad_val)

    trajectories_msip_qg, wts_msip_qg = msip_gs(
        msip_quadgrad,
        N_PARTICLES,
        N_STEPS,
        dim=dimension,
        lr=LR_MSIP,
        init_particles=init_particles,
        kernel_length_scale=SIGMA,
        is_log_density_batched=True,
        kernel_diag_infl=1e-6,
        bounds=(-100.0, 100.0),
        keep_all=True,
        compile_step=True,
        verbose=True,
    )

    # Run deep ensemble
    trajectories_de = deepensembles(
        obj_de,
        N_PARTICLES,
        N_STEPS,
        dimension,
        LR_DE,
        seed=None,
        device=None,
        init_particles=init_particles,
        kernel_length_scale=SIGMA,
        keep_all=True,
        is_log_density_batched=True,
        verbose=True,
    )

    trajectories_dict = {"MSIP": trajectories_msip_qg, "DE": trajectories_de}
    objective_fns_dict = {"MSIP": obj_msip, "DE": obj_de}

    # Optimization diagnostics
    eval_function_trajectories(obj_msip, trajectories_msip_qg, "MSIP")
    eval_function_trajectories(obj_de, trajectories_de, "DE")

    # Visualization
    plot_boundaries(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_mean_prediction(trajectories_dict, objective_fns_dict, X_train, Y_train)
    plot_diversity_curve(trajectories_dict)

    # Evaluation on a dataset
    evaluate(trajectories_dict, objective_fns_dict, X_train, Y_train, "train")
    evaluate(trajectories_dict, objective_fns_dict, X_test, Y_test, "test")
