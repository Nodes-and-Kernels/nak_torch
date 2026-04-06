import torch
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional, Protocol
from dataclasses import dataclass

BatchType = Float[Tensor, "batch"]
PtType = Float[Tensor, " d"]
BatchPtType = Float[Tensor, "batch d"]
QuadrulePtType = Float[Tensor, "quad d"]
QuadruleWtType = Float[Tensor, "quad"]
BatchQuadrulePtType = Float[Tensor, "batch quad d"]
BatchQuadruleWtType = Float[Tensor, "batch quad"]
KernelMatrixType = Float[Tensor, "batch batch"]
GradKernelMatrixType = Float[Tensor, "batch batch d"]

MSIPEstimatorOutput = tuple[BatchType, BatchPtType]

KernelFunction = Callable[[PtType, PtType, float], Float]


class MatSelfKernelFunction(Protocol):
    def __call__(
        self,
        pts: BatchPtType,
        kernel_length_scale: float,
        pts2: Optional[BatchPtType] = None,
    ) -> KernelMatrixType: ...


LogDensity = Callable[[PtType], Float]

GradLogDensity = Callable[[PtType], PtType]

BatchLogDensity = Callable[[BatchPtType], BatchType]

BatchLogDensityGradVal = Callable[[BatchPtType], tuple[BatchPtType, BatchType]]

BatchGradLogDensity = Callable[[BatchPtType], BatchPtType]

BatchQuadratureRule = Callable[[int], tuple[BatchQuadrulePtType, BatchQuadruleWtType]]

ForwardModel = Callable[[Float[Tensor, " dim"]], Float[Tensor, " obs"]]

BatchForwardModel = Callable[[Float[Tensor, "batch dim"]], Float[Tensor, "batch obs"]]


@dataclass
class GaussianModel:
    forward_model: BatchForwardModel
    likelihood_precision: float | Float[Tensor, "obs obs"]
    prior_precision: float | Float[Tensor, "dim dim"]
    true_obs: Float | Float[Tensor, " obs"]
    prior_mean: float | Float[Tensor, " dim"]

    def __init__(
        self,
        forward_model: ForwardModel | BatchForwardModel,
        likelihood_precision: float | Float[Tensor, "obs obs"] = 1.0,
        prior_precision: float | Float[Tensor, "dim dim"] = 1.0,
        true_obs: Float | Float[Tensor, " obs"] = torch.zeros(()),
        prior_mean: float | Float[Tensor, " dim"] = 0.0,
        is_vectorized: bool = False,
    ):
        if is_vectorized:
            self.forward_model = forward_model
        else:
            self.forward_model = torch.vmap(forward_model)
        if prior_mean != 0.0:
            raise ValueError("Only support zero prior mean for now")
        self.likelihood_precision = likelihood_precision
        self.prior_precision = prior_precision
        self.true_obs = true_obs
        self.prior_mean = prior_mean


def gaussian_log_dens_factory(
    model: GaussianModel, compile: bool = True
) -> BatchLogDensity:
    def log_dens(pts: BatchPtType) -> BatchType:
        model_eval = model.forward_model(pts)
        obs_error = model_eval.sub_(model.true_obs)
        like_term = torch.square(torch.linalg.norm(obs_error, dim=-1)).mul_(
            model.likelihood_precision
        )
        like_term.mul_(model.likelihood_precision)
        prior_diff = pts - model.prior_mean
        prior_term = torch.square(torch.linalg.norm(prior_diff, dim=-1)).mul_(
            model.prior_precision
        )
        return -0.5 * (prior_term + like_term)

    return torch.compile(log_dens) if compile else log_dens
