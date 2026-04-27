from typing import Optional

import stan.model
import torch

from nak_torch.tools.types import BatchPtType, BatchType, NAKTarget


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

    def log_dens_batch(self, theta: BatchPtType, _) -> BatchType:
        device, dtype = theta.device, theta.dtype
        out_np = self.model.log_prob(theta.cpu().numpy())
        return torch.as_tensor(out_np, device=device, dtype=dtype)

    def grad_log_dens_batch(self, theta: BatchPtType, _) -> BatchPtType:
        device, dtype = theta.device, theta.dtype
        out_np = self.model.grad_log_prob(theta.cpu().numpy())
        return torch.as_tensor(out_np, device=device, dtype=dtype)

    def grad_val_log_dens_batch(
        self, theta: BatchPtType, _
    ) -> tuple[BatchPtType, BatchType]:
        device, dtype = theta.device, theta.dtype
        out_grad_np, out_val_np = self.model.grad_val_log_prob(theta.cpu().numpy())
        return torch.as_tensor(
            out_grad_np, device=device, dtype=dtype
        ), torch.as_tensor(out_val_np, device=device, dtype=dtype)
