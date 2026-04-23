from typing import Any, Optional

from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor

from nak_torch.tools.util import initialize_particles
from nak_torch.tools.types import (
    BatchTargetEvaluator,
)

from nak_torch.tools.func import (
    GeneralAdaptiveNAKAlgorithm,
)


def nak(
    target: BatchTargetEvaluator,
    algorithm: GeneralAdaptiveNAKAlgorithm,
    n_particles: int,
    n_steps: int,
    lr: float,
    seed: Optional[int] = None,
    init_particles: Optional[Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    target_args: Any = None,
    verbose: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    TODO: Document
    """

    if n_steps < 0:
        raise ValueError("Expected positive number of steps.")

    if seed is not None:
        torch.manual_seed(seed)

    dim, device, dtype = algorithm.dim, algorithm.device, algorithm.dtype
    particles = initialize_particles(
        n_particles, dim, init_particles, device, dtype, bounds
    )

    particle_wts, algorithm_args = algorithm.initialize(particles, target, target_args)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=dtype
        )
        trajectories[0].copy_(particles)
        if algorithm.is_weighted():
            traj_wts = torch.empty(
                (n_steps + 1, particles.shape[0]), device=device, dtype=dtype
            )
            traj_wts[0].copy_(particle_wts)
        else:
            traj_wts = torch.empty(())
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(())

    for idx in tqdm(range(n_steps), disable=not verbose):
        if keep_all:
            trajectories[idx + 1].copy_(particles)
            if algorithm.is_weighted() and keep_all:
                traj_wts[idx + 1].copy_(particle_wts)

        particles, particle_wts, algorithm_args = algorithm.step(
            lr, particles, algorithm_args, target, target_args
        )

        if bounds is not None:
            particles.clamp_(bounds[0], bounds[1])

    if not keep_all:
        trajectories = particles.unsqueeze_(0)
        if algorithm.is_weighted():
            traj_wts = particle_wts.unsqueeze_(0)

    if algorithm.is_weighted():
        return trajectories.detach(), traj_wts.detach()

    return trajectories.detach()
