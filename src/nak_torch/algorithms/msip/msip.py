from dataclasses import astuple, dataclass
from typing import Generic, Optional, TypeVar

import torch

from nak_torch.tools.func import WeightedAdaptiveNAKAlgorithm
from nak_torch.tools.kernel import default_kernel_matrix
from nak_torch.tools.util import quantile_distance
from .msip_map import MSIPEstimatorOutput, msip_map, get_msip_wts
from .estimators import MSIPEstimator
from nak_torch.tools.types import (
    BatchPtType,
    KernelMatrixType,
    MatSelfKernelFunction,
)

MSIPEstimatorOutputT = TypeVar("MSIPEstimatorOutputT", bound=MSIPEstimatorOutput)


@dataclass
class MSIPAlgorithmArgs(Generic[MSIPEstimatorOutputT]):
    kernel_lengthscale: float
    kernel_matrix_inverse: KernelMatrixType
    msip_estimator_output: MSIPEstimatorOutputT


class MSIP(WeightedAdaptiveNAKAlgorithm[MSIPEstimator, MSIPAlgorithmArgs]):
    kernel_diag_infl: float
    default_kernel_lengthscale: float
    kernel_lengthscale_quantile: Optional[float]
    get_kernel_matrix: MatSelfKernelFunction

    def __init__(
        self,
        *_,
        kernel_diag_infl: float = 0.0,
        kernel_lengthscale: Optional[float] = None,
        kernel_lengthscale_quantile: Optional[float] = None,
        get_kernel_matrix: Optional[MatSelfKernelFunction] = None,
    ):
        self.kernel_diag_infl = kernel_diag_infl
        if kernel_lengthscale is None and kernel_lengthscale_quantile is None:
            raise ValueError(
                "Must have either kernel_lengthscale"
                "or kernel_lengthscale_quantile as value"
            )
        self.kernel_lengthscale = kernel_lengthscale
        self.kernel_lengthscale_quantile = kernel_lengthscale_quantile
        if get_kernel_matrix is None:
            self.get_kernel_matrix = default_kernel_matrix
        else:
            self.get_kernel_matrix = get_kernel_matrix

    def get_adaptive_lengthscale(self, particles: BatchPtType) -> float:
        q = self.kernel_lengthscale_quantile
        if q is None:
            return self.default_kernel_lengthscale
        return quantile_distance(particles, q)

    def get_infl_kernel_matrix(self, particles, kernel_lengthscale) -> KernelMatrixType:
        kernel_matrix = self.get_kernel_matrix(particles, kernel_lengthscale)
        if self.kernel_diag_infl is not None:
            kernel_matrix[
                torch.arange(self.n_particles, device=self.device),
                torch.arange(self.n_particles, device=self.device),
            ] += self.kernel_diag_infl
        return kernel_matrix

    def initialize(self, init_particles, target, target_args):
        kernel_lengthscale = self.get_adaptive_lengthscale(init_particles)
        estimator_output = target(init_particles, kernel_lengthscale, target_args)
        kernel_matrix = self.get_infl_kernel_matrix(init_particles, kernel_lengthscale)
        wts = get_msip_wts(init_particles, estimator_output, kernel_matrix)
        return wts, MSIPAlgorithmArgs(
            kernel_lengthscale, kernel_matrix, estimator_output
        )

    def step(self, lr, particles, target, algorithm_args, target_args):
        kernel_lengthscale, kernel_matrix, estimator_output = astuple(algorithm_args)
        kernel_matrix_inverse = torch.linalg.pinv(kernel_matrix)

        # Update the particles
        particles_diff = msip_map(
            estimator_output,
            particles,
            kernel_matrix_inverse,
            output_idx=None,
        )
        new_particles = particles * (1 - lr) + lr * particles_diff

        # Update the parameters
        kernel_lengthscale = self.get_adaptive_lengthscale(new_particles)
        kernel_matrix = self.get_infl_kernel_matrix(new_particles, kernel_lengthscale)
        msip_estimator_output = target(particles, kernel_lengthscale, target_args)
        algorithm_args = MSIPAlgorithmArgs(
            kernel_lengthscale, kernel_matrix_inverse, msip_estimator_output
        )
        new_weights = get_msip_wts(new_particles, estimator_output, kernel_matrix)
        return new_particles, new_weights, algorithm_args
