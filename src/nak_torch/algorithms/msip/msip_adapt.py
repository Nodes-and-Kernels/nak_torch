import warnings
from typing import Callable, Optional

from tqdm import tqdm
import numpy as np
import torch

from nak_torch.tools.kernel import default_kernel_matrix
from nak_torch.tools.util import initialize_particles, get_keywords, quantile_distance
from .msip_map import msip_map
from .estimators import MSIPEstimator, MSIPFredholm
from nak_torch.tools.adaptive_step import default_particle_integrator

from nak_torch.tools.types import (
    BatchPtType,
    LogDensity,
    BatchLogDensity,
    BatchLogDensityGradVal,
    BatchType,
    MatSelfKernelFunction,
)


def process_msip_density(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    *_,
    is_log_density_batched: bool = False,
    gradient_decay: float = 1.0,
    **__,
) -> MSIPEstimator:
    if isinstance(log_density, MSIPEstimator):
        return log_density
    log_density_grad_val: BatchLogDensityGradVal
    if is_log_density_batched:

        def dens_eval(_p):
            out = log_density(_p)
            return out.sum(), out

        log_density_grad_val = torch.func.grad(dens_eval, has_aux=True)
    else:
        log_density_grad_val = torch.vmap(torch.func.grad_and_value(log_density))
    return MSIPFredholm(gradient_decay, log_density_grad_val)


msip_map_used_keys = get_keywords(msip_map) + get_keywords(process_msip_density)


def create_msip_diff(
    get_kernel_matrix: MatSelfKernelFunction,
    kernel_diag_infl: float,
    msip_estimator: MSIPEstimator,
):
    def msip_diff(_, particles: BatchPtType, args):
        (kernel_length_scale,) = args
        n_particles = particles.shape[0]

        kernel_matrix = get_kernel_matrix(particles, kernel_length_scale)
        kernel_matrix[torch.arange(n_particles), torch.arange(n_particles)] += (
            kernel_diag_infl
        )

        msip_estimator_out = msip_estimator.get_v_evals(particles, kernel_length_scale)
        if kernel_diag_infl > 0:
            kernel_matrix_inverse = torch.linalg.inv(kernel_matrix)
        else:
            kernel_matrix_inverse = torch.linalg.pinv(kernel_matrix)

        particles_diff = msip_map(
            msip_estimator_out,
            particles,
            kernel_matrix_inverse,
            output_idx=None,
        )

        return particles_diff - particles

    return torch.compile(msip_diff)


def msip_adapt(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr0: float,
    kernel_length_scale: float,
    noise: float = 0.05,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    get_kernel_matrix: Optional[MatSelfKernelFunction] = None,
    kernel_diag_infl: float = 0.0,
    verbose: bool = False,
    use_quantile_length_scale: Optional[float] = None,
    choose_running: Optional[Callable[[BatchPtType, BatchType], BatchType]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    **msip_kwargs,
):
    r"""
    TODO: Document
    """

    if n_steps < 0:
        raise ValueError("Expected positive number of steps.")

    unused_kwargs = {
        k: v for (k, v) in msip_kwargs.items() if k not in msip_map_used_keys
    }

    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs: {}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)
    if choose_running is None:

        def _choose_running(_: BatchPtType, running: BatchType):
            return running

        choose_running = _choose_running

    if get_kernel_matrix is None:
        get_kernel_matrix = default_kernel_matrix

    msip_estimator = process_msip_density(log_density, **msip_kwargs)
    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), dtype=particles.dtype, device=device
        )
        trajectories[0].copy_(particles)
        dts = torch.empty(
            (n_steps + 1, n_particles), dtype=particles.dtype, device=device
        )
        dts[0] = lr0
    else:
        trajectories = torch.empty(())
        dts = torch.empty(())

    msip_step = create_msip_diff(get_kernel_matrix, kernel_diag_infl, msip_estimator)

    step_fcn, state, running, dt, t = default_particle_integrator(
        particles, msip_step, lr0, rtol=rtol, atol=atol, args=(kernel_length_scale,)
    )
    accepts = torch.zeros(n_particles, dtype=torch.int, device=device)

    for idx in tqdm(range(n_steps), disable=not verbose):
        if use_quantile_length_scale is not None:
            kernel_length_scale = quantile_distance(
                particles, use_quantile_length_scale
            )

        with torch.no_grad():
            dt, t, state, particles, accept = step_fcn(
                state, running, dt, t, particles, (kernel_length_scale,)
            )
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])

        running = choose_running(particles, running)
        if keep_all:
            trajectories[idx + 1].copy_(particles)
            dts[idx + 1].copy_(dt)

        accepts += accept

    if not keep_all:
        trajectories = particles.unsqueeze_(0)

    return trajectories.detach(), (dts.detach(), accepts.detach())
