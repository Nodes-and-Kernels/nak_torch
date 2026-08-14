from dataclasses import dataclass
from typing import Generic, NamedTuple, Optional, TypeVar

import torch

from nak_torch.algorithms.msip.msip_map import msip_map_one_output
from nak_torch.tools.func import AlgorithmArgsT, WeightedAdaptiveNAKAlgorithm
from nak_torch.tools.kernel import DEFAULT_KERNEL_MATRIX
from nak_torch.tools.util import get_keywords, quantile_distance
from .estimators import MSIPEstimator, MSIPFredholm

from nak_torch.tools.types import (
    BatchPtType,
    DeviceLike,
    KernelMatrixType,
    LogDensity,
    BatchLogDensity,
    BatchLogDensityGradVal,
    MSIPEstimatorOutput,
    MatSelfKernelFunction,
)


MSIPEstimatorOutputT = TypeVar("MSIPEstimatorOutputT", bound=MSIPEstimatorOutput)
MSIPAlgorithmArgsT = TypeVar("MSIPAlgorithmArgsT")


class MSIPAlgorithmArgs(NamedTuple, Generic[MSIPEstimatorOutputT]):
    kernel_lengthscale: float
    kernel_matrix: KernelMatrixType
    msip_estimator_output: MSIPEstimatorOutputT


@dataclass
class MSIPGSAlgorithmArgs(Generic[MSIPEstimatorOutputT]):
    kernel_lengthscale: float
    msip_estimator_output: MSIPEstimatorOutputT


class GeneralMSIPAlgorithm(WeightedAdaptiveNAKAlgorithm[MSIPEstimator, AlgorithmArgsT]):
    kernel_diag_infl: Optional[float]
    default_kernel_lengthscale: float
    kernel_lengthscale_quantile: Optional[float]
    get_kernel_matrix: MatSelfKernelFunction

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[torch.dtype] = None,
        *_,
        kernel_diag_infl: Optional[float] = None,
        kernel_lengthscale: Optional[float] = None,
        kernel_lengthscale_quantile: Optional[float] = None,
        get_kernel_matrix: Optional[MatSelfKernelFunction] = None,
        **kwargs,
    ):
        super().__init__(dim, n_particles, device, dtype, **kwargs)
        self.kernel_diag_infl = kernel_diag_infl
        if kernel_lengthscale is None and kernel_lengthscale_quantile is None:
            raise ValueError(
                "Must have either kernel_lengthscale "
                "or kernel_lengthscale_quantile as value"
            )
        if kernel_lengthscale is None:
            self.default_kernel_lengthscale = 0.0
        else:
            self.default_kernel_lengthscale = kernel_lengthscale
        self.kernel_lengthscale_quantile = kernel_lengthscale_quantile
        if get_kernel_matrix is None:
            self.get_kernel_matrix = DEFAULT_KERNEL_MATRIX
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
            kernel_matrix += self.kernel_diag_infl * torch.eye(
                kernel_matrix.shape[0],
                dtype=kernel_matrix.dtype,
                device=kernel_matrix.device,
            )
        return kernel_matrix


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

        def dens_eval(_p, target_args):
            out = log_density(_p, target_args)
            return out.sum(), out

        log_density_grad_val = torch.func.grad(dens_eval, has_aux=True)
    else:
        log_density_grad_val = torch.vmap(torch.func.grad_and_value(log_density))
    return MSIPFredholm(gradient_decay, log_density_grad_val)


msip_map_used_keys = get_keywords(msip_map_one_output) + get_keywords(
    process_msip_density
)
