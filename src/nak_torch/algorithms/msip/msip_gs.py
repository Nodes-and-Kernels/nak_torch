import warnings
from typing import Optional

from tqdm import tqdm
import numpy as np
import torch

from nak_torch.tools.kernel import default_kernel_matrix
from nak_torch.tools.util import initialize_particles, get_keywords, quantile_distance
from .msip_map import MSIPEstimatorOutput, msip_map, get_msip_wts
from .estimators import MSIPEstimator, MSIPFredholm

from nak_torch.tools.types import LogDensity, BatchLogDensity, \
    BatchLogDensityGradVal, BatchType, MatSelfKernelFunction


# Gauss-Seidel variant of MSIP.

def process_msip_density(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    *_,
    is_log_density_batched: bool = False,
    gradient_decay: float = 1.0,
    **__
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
        log_density_grad_val = torch.vmap(
            torch.func.grad_and_value(log_density))
    return MSIPFredholm(gradient_decay, log_density_grad_val)


msip_map_used_keys = get_keywords(msip_map) + \
    get_keywords(process_msip_density)


def msip_gs(
    log_density: LogDensity | BatchLogDensity | MSIPEstimator,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
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
    compile_step: bool = True,
    **msip_kwargs
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
    if get_kernel_matrix is None:
        get_kernel_matrix = default_kernel_matrix

    msip_estimator = process_msip_density(log_density, **msip_kwargs)
    est_v = msip_estimator.get_v_evals
    _msip_map = msip_map
    _get_msip_wts = get_msip_wts
    if compile_step:
        _msip_map = torch.compile(msip_map)
        _get_msip_wts = torch.compile(_get_msip_wts)
        est_v = torch.compile(est_v)

    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps+1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
        traj_wts = torch.empty(
            (n_steps+1, particles.shape[0]), device=device, dtype=particles.dtype
        )
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(())

    msip_estimator_out: MSIPEstimatorOutput
    particle_wts: BatchType
    
    for step in tqdm(range(n_steps + 1), disable=not verbose):

        if use_quantile_length_scale is not None:
            kernel_length_scale = quantile_distance(particles, use_quantile_length_scale)
    
        for i in range(n_particles):
            ls_i = (quantile_distance(particles, use_quantile_length_scale)
                    if use_quantile_length_scale is not None else kernel_length_scale)
    
            km_i = get_kernel_matrix(particles, ls_i)
            km_i[torch.arange(n_particles), torch.arange(n_particles)] += kernel_diag_infl
    
            est_out_i = est_v(particles, ls_i)
    
            particle_wts = _get_msip_wts(particles, est_out_i, km_i)
    
            if keep_all and i == n_particles - 1:
                traj_wts[step].copy_(particle_wts)
    
            if step >= n_steps:
                continue  
    
            if kernel_diag_infl > 0:
                km_inv_i = torch.linalg.inv(km_i)
            else:
                km_inv_i = torch.linalg.pinv(km_i)
    
            target_i = _msip_map(est_out_i, particles, km_inv_i, output_idx=i)
    
            with torch.no_grad():
                particles[i] = (1.0 - lr) * particles[i] + lr * target_i
                if bounds is not None:
                    particles[i].clamp_(bounds[0], bounds[1])
    
        if keep_all and step < n_steps:
            trajectories[step + 1].copy_(particles)
    
    

    if not keep_all:
        trajectories = particles.unsqueeze_(0)
        traj_wts = particle_wts.unsqueeze_(0) 

    return trajectories.detach(), traj_wts.detach()
