import warnings
from typing import Optional

from tqdm import tqdm
import numpy as np
import torch

from nak_torch.tools.kernel import default_kernel_matrix
from nak_torch.tools.util import initialize_particles, quantile_distance
from .msip_map import msip_map, get_msip_wts
from .estimators import MSIPEstimator
from .msip_tools import msip_map_used_keys, process_msip_density

from nak_torch.tools.types import (
    LogDensity,
    BatchLogDensity,
    BatchType,
    MatSelfKernelFunction,
)


# Gauss-Seidel variant of MSIP.
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

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
        traj_wts = torch.empty(
            (n_steps + 1, particles.shape[0]), device=device, dtype=particles.dtype
        )
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(())

    particle_wts: BatchType = torch.tensor(())

    if use_quantile_length_scale is not None:
        kernel_length_scale = quantile_distance(particles, use_quantile_length_scale)
    est_out = est_v(particles, kernel_length_scale)

    # est_out should keep references to est_out_0 and est_out_1
    est_out_0, est_out_1 = est_out
    for step in tqdm(range(n_steps + 1), disable=not verbose):
        for i in range(n_particles):
            if use_quantile_length_scale is not None:
                kernel_length_scale = quantile_distance(
                    particles, use_quantile_length_scale
                )

            km_i = get_kernel_matrix(particles, kernel_length_scale)
            if kernel_diag_infl > 0:
                km_i[torch.arange(n_particles), torch.arange(n_particles)] += (
                    kernel_diag_infl
                )

            est_out_i_0, est_out_i_1 = est_v(
                particles[i].unsqueeze(0), kernel_length_scale
            )
            est_out_0[i].copy_(est_out_i_0.squeeze())
            est_out_1[i].copy_(est_out_i_1.squeeze())

            particle_wts = _get_msip_wts(particles, est_out, km_i)

            if keep_all and i == n_particles - 1:
                traj_wts[step].copy_(particle_wts)

            if step >= n_steps:
                continue

            if kernel_diag_infl > 0:
                km_inv_i = torch.linalg.inv(km_i)
            else:
                km_inv_i = torch.linalg.pinv(km_i)

            target_i = _msip_map(est_out, particles, km_inv_i, output_idx=i)

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
