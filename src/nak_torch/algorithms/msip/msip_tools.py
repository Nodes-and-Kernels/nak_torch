import torch

from nak_torch.tools.util import get_keywords
from .msip_map import msip_map
from .estimators import MSIPEstimator, MSIPFredholm

from nak_torch.tools.types import (
    LogDensity,
    BatchLogDensity,
    BatchLogDensityGradVal,
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
