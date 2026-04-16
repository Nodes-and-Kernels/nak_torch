from typing import Optional

import torch

from nak_torch.tools import recursive_weighted_average_alpha_v
from nak_torch.tools.kernel import kernel_optimal_weight_factory
from nak_torch.tools.types import (
    KernelMatrixType,
    PtType,
    BatchPtType,
    BatchType,
    MSIPEstimatorOutput,
)


def calculate_msip_map(
    K_minus_one_i: BatchType,
    particles: BatchPtType,
    eval_log_v0: BatchType,
    grad_log_v0: BatchPtType,
) -> BatchPtType:

    term_v0, _ = recursive_weighted_average_alpha_v(
        particles, K_minus_one_i, log_v=eval_log_v0
    )
    term_v1, _ = recursive_weighted_average_alpha_v(
        grad_log_v0, K_minus_one_i, log_v=eval_log_v0
    )
    return term_v0 + term_v1


calculate_msip_map_all = torch.vmap(calculate_msip_map, in_dims=(0, None, None, None))


def get_msip_wts(
    particles: BatchPtType,
    msip_estimators: MSIPEstimatorOutput,
    kernel_matrix: KernelMatrixType,
) -> BatchType:
    log_v0 = msip_estimators[0]
    return kernel_optimal_weight_factory(particles, log_v0, kernel_matrix)


def msip_map(
    estimators: MSIPEstimatorOutput,
    particles: torch.Tensor,
    kernel_matrix_inverse: KernelMatrixType,
    output_idx: Optional[int],
) -> PtType | BatchPtType:
    """
    Compute the full MSIP map T(y) for all particles at once.
    Returns t_arr with shape (N, d).
    """
    particles = particles.clone()

    with torch.no_grad():
        N, d = particles.shape

        if output_idx is None:
            return calculate_msip_map_all(kernel_matrix_inverse, particles, *estimators)
        else:
            return calculate_msip_map(
                kernel_matrix_inverse[output_idx],
                particles,
                *estimators,
            )
