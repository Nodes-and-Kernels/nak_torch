#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from typing import Optional, Callable
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem, kernel_grad_and_value_factory
from nak_torch.tools.types import KernelFunction, BatchGradLogDensity, BatchPtType
from nak_torch.tools.util import (
    batched_grad_log_density_factory,
    initialize_particles,
    quantile_distance,
)


def create_svgd_step(
    kernel_elem: KernelFunction, grad_log_p: BatchGradLogDensity
) -> Callable[[BatchPtType, float], BatchPtType]:
    which_argnum = 1
    kernel_grad_val = kernel_grad_and_value_factory(
        kernel_elem,
        which_argnum,
    )

    def svgd_step_dir(points: BatchPtType, kernel_lengthscale: float):
        # ASSUME SYMMETRY OF KERNEL
        # kg[i,j,ell] = grad(x_j[ell]) k(x_i, x_j), k[i,j] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val(points, points, kernel_lengthscale)
        # lpg[j,ell] = grad(x_j[ell]) log_p(x_j)
        log_p_grad_ev = grad_log_p(points)
        # term_1[i, ell] = sum_j k(i, j) grad(x_j[ell]) log_p(x_j)
        term_1 = k_eval @ log_p_grad_ev
        # term_2[i, ell] = sum_j grad(x_j[ell]) k(x_i, x_j)
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return svgd_step_dir


def svgd(
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
    use_quantile_length_scale: Optional[float] = None,
    grad_log_density: Optional[BatchGradLogDensity] = None,
    verbose: bool = False,
    fudge_factor: float = 1e-6,
    grad_decay_factor: float = 0.9,
    **unused_kwargs,
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)

    if use_quantile_length_scale is not None and (
        use_quantile_length_scale > 1.0 or use_quantile_length_scale < 0.0
    ):
        raise ValueError(
            f"Expected use_quantile_length_scale in [0,1], got {use_quantile_length_scale}"
        )

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    grad_log_p = batched_grad_log_density_factory(
        log_density, is_log_density_batched, grad_log_density
    )
    step_fcn = create_svgd_step(kernel_elem, grad_log_p)

    historical_grad: Optional[BatchPtType] = None

    for idx in tqdm(range(n_steps), disable=not verbose):
        if use_quantile_length_scale is not None:
            kernel_length_scale = quantile_distance(
                particles, use_quantile_length_scale
            )

        particles_diff = step_fcn(particles, kernel_length_scale)
        if historical_grad is None:
            historical_grad = particles_diff.square()
        else:
            decay_add = particles_diff.square().mul_(1 - grad_decay_factor)
            historical_grad = historical_grad.mul_(grad_decay_factor).add_(decay_add)
        particles_diff = particles_diff.div_(historical_grad.sqrt().add_(fudge_factor))
        with torch.no_grad():
            particles = particles + lr * particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx + 1].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
