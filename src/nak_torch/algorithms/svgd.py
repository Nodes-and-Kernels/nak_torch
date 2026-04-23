#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

from dataclasses import astuple, dataclass
from typing import Optional, Callable
import torch
from nak_torch.tools.func import UnweightedAdaptiveNAKAlgorithm
from nak_torch.tools.kernel import kernel_grad_and_value_factory, default_kernel_elem
from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchKernelGradValFunction,
    DeviceLike,
    KernelFunction,
    BatchGradLogDensity,
    BatchPtType,
)
from nak_torch.tools.util import quantile_distance


def create_svgd_step(
    kernel_elem: KernelFunction, grad_log_p: BatchGradLogDensity, *kernel_elem_args
) -> Callable[[BatchPtType], BatchPtType]:
    which_argnum = 1
    kernel_grad_val = kernel_grad_and_value_factory(
        kernel_elem, which_argnum, *kernel_elem_args
    )

    def svgd_step_dir(points: BatchPtType):
        # ASSUME SYMMETRY OF KERNEL
        # kg[i,j,ell] = grad(x_j[ell]) k(x_i, x_j), k[i,j] = k(x_i, x_j)
        k_grad, k_eval = kernel_grad_val(points, points)
        # lpg[j,ell] = grad(x_j[ell]) log_p(x_j)
        log_p_grad_ev = grad_log_p(points)
        # term_1[i, ell] = sum_j k(i, j) grad(x_j[ell]) log_p(x_j)
        term_1 = k_eval @ log_p_grad_ev
        # term_2[i, ell] = sum_j grad(x_j[ell]) k(x_i, x_j)
        term_2 = k_grad.sum(1)
        return (term_1 + term_2) / points.shape[0]

    return svgd_step_dir


def create_svgd_kernel_grad_val(
    kernel_elem: KernelFunction,
) -> BatchKernelGradValFunction:
    which_argnum = 1
    kernel_grad_val = torch.func.grad_and_value(kernel_elem, argnums=which_argnum)
    kernel_grad_val_vec = torch.vmap(
        torch.vmap(kernel_grad_val, in_dims=(None, 0, None)), in_dims=(0, None, None)
    )
    return kernel_grad_val_vec


def svgd_step(
    kernel_grad_val: BatchKernelGradValFunction,
    points: BatchPtType,
    grad_log_dens: BatchPtType,
    kernel_elem_args,
) -> BatchPtType:
    k_grad, k_eval = kernel_grad_val(points, points, kernel_elem_args)
    # lpg[j,ell] = grad(x_j[ell]) log_p(x_j)
    log_p_grad_ev = grad_log_dens
    # term_1[i, ell] = sum_j k(i, j) grad(x_j[ell]) log_p(x_j)
    term_1: BatchPtType = k_eval @ log_p_grad_ev
    # term_2[i, ell] = sum_j grad(x_j[ell]) k(x_i, x_j)
    term_2: BatchPtType = k_grad.sum(1)
    return (term_1 + term_2) / points.shape[0]


@dataclass
class SVGDAlgorithmArgs:
    kernel_lengthscale: float


class SVGDAlgorithm(
    UnweightedAdaptiveNAKAlgorithm[BatchGradLogDensityEvaluator, SVGDAlgorithmArgs]
):
    default_kernel_lengthscale: float
    kernel_lengthscale_quantile: Optional[float]
    kernel_grad_val: BatchKernelGradValFunction

    def get_adaptive_lengthscale(self, particles: BatchPtType) -> float:
        q = self.kernel_lengthscale_quantile
        if q is None:
            return self.default_kernel_lengthscale
        return quantile_distance(particles, q)

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[torch.dtype] = None,
        *_,
        default_kernel_lengthscale: Optional[float] = None,
        kernel_lengthscale_quantile: Optional[float] = None,
        kernel_elem: Optional[KernelFunction] = None,
    ):
        super().__init__(dim, n_particles, device, dtype)
        if default_kernel_lengthscale is None and kernel_lengthscale_quantile is None:
            raise ValueError(
                "Must provide either default_kernel_lengthscale or kernel_lengthscale_quantile"
            )
        if kernel_lengthscale_quantile is not None and (
            kernel_lengthscale_quantile < 0 or kernel_lengthscale_quantile > 1
        ):
            raise ValueError(
                f"Expected kernel_lengthscale_quantile in [0,1], given {kernel_lengthscale_quantile}"
            )
        if default_kernel_lengthscale is None:
            default_kernel_lengthscale = 0.0
        if kernel_elem is None:
            kernel_elem = default_kernel_elem
        self.default_kernel_lengthscale = default_kernel_lengthscale
        self.kernel_lengthscale_quantile = kernel_lengthscale_quantile
        self.kernel_grad_val = create_svgd_kernel_grad_val(kernel_elem)

    def initialize(self, init_particles, target, target_args):
        kernel_lengthscale = self.get_adaptive_lengthscale(init_particles)
        return None, SVGDAlgorithmArgs(kernel_lengthscale)

    def step(self, lr, particles, target, algorithm_args, target_args):
        (kernel_lengthscale,) = astuple(algorithm_args)
        grad_log_dens_eval = target(particles, None, target_args)
        particles_diff = svgd_step(
            self.kernel_grad_val, particles, grad_log_dens_eval, kernel_lengthscale
        )
        new_particles = particles_diff.mul_(lr).add_(particles)
        new_kernel_lengthscale = self.get_adaptive_lengthscale(new_particles)
        return new_particles, None, SVGDAlgorithmArgs(new_kernel_lengthscale)
