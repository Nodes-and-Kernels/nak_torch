from typing import Optional
import re

import stan.model
import torch
from posteriordb.posterior import Posterior

from nak_torch.tools.types import BatchPtType, BatchType, NAKTarget

__all__ = ["get_draws", "StanModel"]


def expanded_var_names(model: stan.model.Model):
    names = []
    array_param = re.compile(r"\.\d+$")
    for var in model.constrained_param_names:
        if array_param.search(var) is None:
            names.append(var)
        else:
            v, n = var.split(".")
            names.append(v + "[" + n + "]")
    return names


def get_draws(model: stan.model.Model, posterior: Posterior):
    reference_draws = posterior.reference_draws()
    var_names = expanded_var_names(model)
    all_draws = torch.concat(
        [
            torch.column_stack([torch.as_tensor(chain[v]) for v in var_names])
            for chain in reference_draws
        ]
    )
    return all_draws


class StanModel(NAKTarget):
    dim: int

    def __init__(self, model: stan.model.Model, dim: Optional[int] = None):
        if dim is None:
            all_dims = model.dims
            if any(len(d) > 1 for d in all_dims):
                raise ValueError(
                    f"Can currently only handle models with one-dimensional variables. Got dims {all_dims}"
                )
            self.dim = sum(1 if len(x) == 0 else x[0] for x in all_dims)
        else:
            self.dim = dim
        self.model = model

    def log_dens_batch(self, theta: BatchPtType, _=None) -> BatchType:
        device, dtype = theta.device, theta.dtype
        out_np = self.model.log_prob(theta.cpu().to(dtype=torch.float64).numpy())
        return torch.as_tensor(out_np, device=device, dtype=dtype)

    def grad_log_dens_batch(self, theta: BatchPtType, _=None) -> BatchPtType:
        device, dtype = theta.device, theta.dtype
        out_np = self.model.grad_log_prob(theta.cpu().to(dtype=torch.float64).numpy())
        return torch.as_tensor(out_np, device=device, dtype=dtype)

    def grad_val_log_dens_batch(
        self, theta: BatchPtType, _=None
    ) -> tuple[BatchPtType, BatchType]:
        device, dtype = theta.device, theta.dtype
        out_grad_np, out_val_np = self.model.grad_val_log_prob(
            theta.cpu().to(dtype=torch.float64).numpy()
        )
        return torch.as_tensor(
            out_grad_np, device=device, dtype=dtype
        ), torch.as_tensor(out_val_np, device=device, dtype=dtype)
