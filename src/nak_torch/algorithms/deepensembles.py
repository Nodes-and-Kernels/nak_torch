#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of deep ensembles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from typing import Optional, Callable
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import KernelFunction, BatchGradLogDensity, BatchPtType
from nak_torch.tools.util import batched_grad_log_density_factory, initialize_particles


def create_deepensembles_step(grad_log_p: BatchGradLogDensity) -> Callable[[BatchPtType], BatchPtType]:
    def deepensembles_step_dir(points: BatchPtType):
        log_p_grad_ev = grad_log_p(points)

        return  log_p_grad_ev

    return deepensembles_step_dir


def deepensembles(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    verbose: bool = False,
    **unused_kwargs
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    grad_log_p = batched_grad_log_density_factory(log_density, is_log_density_batched, grad_log_density)
    step_fcn = create_deepensembles_step(grad_log_p)
    
    trajectories[0].copy_(particles)
    
    for idx in tqdm(range(n_steps-1), disable=not verbose):
        particles_diff = step_fcn(particles)
        with torch.no_grad():
            particles = particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx+1].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
