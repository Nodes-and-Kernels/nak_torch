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
        out = torch.empty(theta.shape[0], dtype=theta.dtype, device="cpu")
        for theta_idx in range(theta.shape[0]):
            th = theta[theta_idx].cpu().tolist()
            out[theta_idx] = self.model.log_prob(th)
        return out.to(device=theta.device)

    def grad_log_dens_batch(self, theta: BatchPtType, _) -> BatchPtType:
        out = torch.empty_like(theta, device="cpu")
        for theta_idx in range(theta.shape[0]):
            th = theta[theta_idx].cpu().tolist()
            out_i = self.model.grad_log_prob(th)
            out[theta_idx] = torch.as_tensor(out_i)
        return out.to(device=theta.device)

    def grad_val_log_dens_batch(
        self, theta: BatchPtType, _
    ) -> tuple[BatchPtType, BatchType]:
        out_grad = torch.empty_like(theta, device="cpu")
        out = torch.empty(theta.shape[0], device="cpu")
        for theta_idx in range(theta.shape[0]):
            th = theta[theta_idx].cpu().tolist()
            out_grad_i = self.model.grad_log_prob(th)
            out_grad[theta_idx] = torch.as_tensor(out_grad_i)
            out[theta_idx] = self.model.log_prob(th)
        return out_grad.to(device=theta.device), out.to(device=theta.device)
