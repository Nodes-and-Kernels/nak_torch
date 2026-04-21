from typing import Optional

import torch
import pyro
import pyro.distributions as dist
from nak_torch import GaussianModel
from nak_torch.tools.types import DeviceLike

def get_pyro_std_from_prec(
    prec: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    nd = prec.ndim
    if nd == 0 or nd == 1:
        return torch.sqrt(1 / prec)
    elif nd == 2:
        diag_prec = torch.diag(prec)
        abs_err_from_diag = torch.linalg.norm(torch.diag(diag_prec) - prec)
        abs_norm = torch.linalg.norm(prec)
        if abs_err_from_diag / abs_norm > 1e-6:
            raise ValueError("Currently support diagonal covariance matrices.")
        return torch.sqrt(1 / diag_prec)
    else:
        raise ValueError(f"Expected precision to be ndim=0,1,2, got {nd}.")


class PyroModel:
    def __init__(
        self,
        model: GaussianModel,
        param_dim: Optional[int] = None,
        device: Optional[DeviceLike] = None,
    ):
        like_prec = torch.as_tensor(model.likelihood_precision, device=device)
        prior_prec = torch.as_tensor(model.prior_precision, device=device)
        self.like_std = get_pyro_std_from_prec(like_prec)
        prior_std = get_pyro_std_from_prec(prior_prec)
        self.forward_model = model.forward_model
        prior_mean = torch.as_tensor(model.prior_mean, device=device)

        if param_dim is not None:
            if prior_mean.numel() == 1:
                self.prior_mean = prior_mean.item() * torch.ones(param_dim)
            elif prior_mean.numel() == param_dim:
                self.prior_mean = prior_mean.flatten()
            else:
                raise ValueError(
                    f"Unexpected arguments: prior_mean.size = {prior_mean.size}, param_dim = {param_dim}"
                )
            if prior_std.numel() == 1:
                self.prior_std = prior_std * torch.ones(param_dim)
            elif prior_std.shape == (param_dim,):
                self.prior_std = prior_std.flatten()
            else:
                raise ValueError(
                    f"Unexpected arguments: prior_std.shape = {prior_std.shape}, param_dim = {param_dim}"
                )

    def __call__(self, data):
        theta = pyro.sample("theta", dist.Normal(self.prior_mean, self.prior_std))
        mean_out = self.forward_model(theta.unsqueeze(0))
        with pyro.plate("data", len(data)):
            return pyro.sample("obs", dist.Normal(mean_out, self.like_std), obs=data)
