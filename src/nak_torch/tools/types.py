from typing import Any, Callable, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from jaxtyping import Float

DeviceLike = str | torch.device | int

BatchType = Float[Tensor, "batch"]
PtType = Float[Tensor, " d"]
BatchPtType = Float[Tensor, "batch d"]
QuadrulePtType = Float[Tensor, "quad d"]
QuadruleWtType = Float[Tensor, "quad"]
BatchQuadrulePtType = Float[Tensor, "batch quad d"]
BatchQuadruleWtType = Float[Tensor, "batch quad"]
KernelMatrixType = Float[Tensor, "batch batch"]
GradKernelMatrixType = Float[Tensor, "batch batch d"]

DensityGradValOutput = tuple[BatchPtType, BatchType]
MSIPEstimatorOutput = tuple[BatchType, BatchPtType]

KernelFunction = Callable[[PtType, PtType, float], Float]

EvaluatorOutput = TypeVar("EvaluatorOutput")


class BatchDensityEvaluator(ABC, Generic[EvaluatorOutput]):
    @abstractmethod
    def __call__(
        self, particles: BatchPtType, evaluator_args, *target_args
    ) -> EvaluatorOutput:
        pass


class MatSelfKernelFunction(Protocol):
    def __call__(
        self,
        pts: BatchPtType,
        kernel_length_scale: float,
        pts2: Optional[BatchPtType] = None,
    ) -> KernelMatrixType: ...


LogDensity = Callable[[PtType, Any], Float]

GradLogDensity = Callable[[PtType, Any], PtType]

LogDensityGradVal = Callable[[PtType, Any], tuple[PtType, Float]]

BatchLogDensity = Callable[[BatchPtType, Any], BatchType]

BatchLogDensityGradVal = Callable[[BatchPtType, Any], DensityGradValOutput]

BatchGradLogDensity = Callable[[BatchPtType, Any], BatchPtType]

BatchQuadratureRule = Callable[[int], tuple[BatchQuadrulePtType, BatchQuadruleWtType]]

ForwardModel = Callable[[Float[Tensor, " dim"], Any], Float[Tensor, " obs"]]

BatchForwardModel = Callable[
    [Float[Tensor, "batch dim"], Any], Float[Tensor, "batch obs"]
]


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
        batch_forward_model: BatchForwardModel
        if is_vectorized:
            batch_forward_model = forward_model  # type: ignore
        else:
            batch_forward_model = torch.vmap(forward_model, in_dims=(0, None))
        self.forward_model = batch_forward_model
        self.prior_mean = prior_mean
        self.likelihood_precision = likelihood_precision
        self.prior_precision = prior_precision
        self.true_obs = true_obs
        self.prior_mean = prior_mean

    def to_log_dens(self, use_compiled: bool = True) -> BatchLogDensity:
        def log_dens(pts: BatchPtType, aux_args: Any) -> BatchType:
            model_eval = self.forward_model(pts, aux_args)
            obs_error = model_eval.sub_(self.true_obs)
            like_term = torch.square(torch.linalg.norm(obs_error, dim=-1)).mul_(
                self.likelihood_precision
            )
            like_term.mul_(self.likelihood_precision)
            prior_diff = pts
            if self.prior_mean != 0.0:
                prior_diff -= self.prior_mean
            prior_term = torch.square(torch.linalg.norm(prior_diff, dim=-1)).mul_(
                self.prior_precision
            )
            return -0.5 * (prior_term + like_term)

        return torch.compile(log_dens) if use_compiled else log_dens
