from typing import Optional, Union

import torch
from torch import Tensor
from jaxtyping import Float
import pyro
import pyro.distributions as dist
from .types import GaussianModel, LogisticRegressionModel, AbstractModel
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

__all__ = ["pyro_model_factory"]

DeviceLike = Union[str, torch.device, int]


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


ModelT = TypeVar("ModelT", bound=AbstractModel)


class PyroModel(ABC, Generic[ModelT]):
    @abstractmethod
    def __init__(
        self,
        model: ModelT,
        param_dim: Optional[int] = None,
        device: Optional[DeviceLike] = None,
    ):
        pass

    @abstractmethod
    def __call__(self, data: Float[Tensor, "batch dim"]) -> Float[Tensor, " batch"]:
        pass


def pyro_model_factory(
    model: AbstractModel,
    param_dim: Optional[int] = None,
    device: Optional[DeviceLike] = None,
):
    match model:
        case GaussianModel():
            return PyroGaussianModel(model, param_dim, device)
        case LogisticRegressionModel():
            return PyroLogisticRegressionModel(model, param_dim, device)
        case _ as unreachable:
            raise ValueError(f"Unexpected model type {unreachable}")


class PyroGaussianModel(PyroModel[GaussianModel]):
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


class PyroLogisticRegressionModel(PyroModel[LogisticRegressionModel]):
    concentration: Float
    rate: Float
    param_dim: int
    prior_mean: Float | Float[Tensor, " dim"]

    def __init__(
        self,
        model: LogisticRegressionModel,
        param_dim: Optional[int] = None,
        device: Optional[DeviceLike] = None,
    ):
        self.concentration, self.rate = (
            model.hyperprior.concentration,
            model.hyperprior.rate,
        )
        prior_mean = model.prior_mean
        if prior_mean is None:
            prior_mean = 0.0
        prior_mean = torch.as_tensor(prior_mean, device=device)
        if param_dim is not None:
            coeff_dim = param_dim - 1
            if prior_mean.numel() == 1:
                self.prior_mean = prior_mean.item() * torch.ones(coeff_dim)
            elif prior_mean.numel() == coeff_dim:
                self.prior_mean = prior_mean.flatten()
            else:
                raise ValueError(
                    f"Unexpected arguments: prior_mean.size = {prior_mean.shape}, coeff_dim = {coeff_dim}"
                )
            self.param_dim = param_dim
        else:
            self.param_dim = prior_mean.shape[0] + 1
            self.prior_mean = prior_mean

    def __call__(self, data):
        # Data should be dimension equiv to all but alpha, plus labels as last row
        if data.shape[0] != (self.param_dim - 1) + 1:
            raise ValueError(
                f"Got data.shape[0] = {data.shape[0]}, expected {self.param_dim}"
            )
        prior_precision = pyro.sample(
            "alpha", dist.Gamma(self.concentration, self.rate)
        )
        prior_std = 1 / prior_precision.sqrt()
        theta = pyro.sample("theta", dist.Normal(self.prior_mean, prior_std))
        dataset, labels = data[:-1], data[-1]
        with pyro.plate("data", dataset.shape[1]):
            logits = theta @ dataset
            return pyro.sample(
                "obs", dist.Bernoulli(logits=logits, validate_args=True), obs=labels
            )
