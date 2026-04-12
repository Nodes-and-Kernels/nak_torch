import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float
from typing import Callable, Optional, Protocol, Self
from dataclasses import dataclass
from abc import ABC, abstractmethod

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


class AbstractModel(ABC):
    @abstractmethod
    def to_log_dens(
        self: Self, use_compiled: bool = True
    ) -> Callable[[BatchPtType], BatchType]:
        pass


@dataclass
class GaussianModel(AbstractModel):
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
        if not is_vectorized:
            forward_model = torch.vmap(forward_model)
        self.forward_model = forward_model
        self.prior_mean = prior_mean
        self.likelihood_precision = likelihood_precision
        self.prior_precision = prior_precision
        self.true_obs = true_obs
        self.prior_mean = prior_mean

    def to_log_dens(self, use_compiled=True):
        return gaussian_log_dens_factory(self, use_compiled)


def bernoulli_loglikelihood_logit(logits, labels):
    # If logit(p) = log(p / (1-p)), bernoulli log-likelihood is
    # log pi(y | p) = y*logit(p) + log(1-p)
    # If q = logit(p), then log(1-p) = -softplus(q) and
    # log pi(y | q) = y * q - softplus(q)
    constant_term = -torch.nn.functional.softplus(logits)
    return torch.sum(labels * logits + constant_term)


bernoulli_loglikelihood_logit_v = torch.vmap(bernoulli_loglikelihood_logit, (0, None))


@dataclass
class LogisticRegressionModel(AbstractModel):
    """Assumes a gaussian prior and linear model for logits"""

    dim: int
    prior_mean: float | Float[Tensor, " dim"] | None
    data: Float | Float[Tensor, "dim labels"]
    labels: Float | Float[Tensor, " labels"]
    hyperprior: torch.distributions.Gamma

    def __init__(
        self,
        data_or_fname: Float[Tensor, "dim-1 labels"] | str,
        labels: Optional[Float[Tensor, " labels"]],
        prior_mean: float | Float[Tensor, " dim"] | None = None,
        dtype=None,
        device=None,
        hyperprior_a=1.0,
        hyperprior_b=0.1,
    ):
        data: torch.Tensor
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device

        def as_tensor(t):
            return torch.as_tensor(t, dtype=dtype, device=device)

        self.prior_mean = prior_mean if prior_mean is None else as_tensor(prior_mean)
        if isinstance(data_or_fname, str):
            data = as_tensor(np.load(data_or_fname))
            if labels is None:  # Split labels from data
                labels = data[:, -1]
                data = data[:, :-1]
        elif isinstance(data_or_fname, torch.Tensor):
            data = data_or_fname
        else:
            raise ValueError(
                f"Expected data_or_fname to be str or tensor, got {type(data_or_fname)}"
            )
        if labels is None or labels.shape[0] != data.shape[0]:
            raise ValueError("Unexpected type or size of argument `labels`.")
        constant = as_tensor(torch.ones(data.shape[0]))
        self.data = torch.column_stack((constant, data)).T
        self.dim = self.data.shape[0] + 1
        self.labels = labels
        self.prior_mean = prior_mean
        self.hyperprior = torch.distributions.Gamma(
            as_tensor(hyperprior_a), as_tensor(hyperprior_b)
        )

    def to_log_dens(self, use_compiled: bool = True):
        def log_hyperprior(t):
            return self.hyperprior.log_prob(t)

        def log_dens(params: BatchPtType) -> BatchType:
            is_batch = params.ndim == 2
            if not is_batch:
                params = params.unsqueeze(0)
            if params.shape[1] != self.dim:
                raise ValueError(
                    f"Got params.shape[1] = {params.shape[1]}, expected {self.dim}"
                )
            prior_diff = params.clone()
            if self.prior_mean is not None:
                prior_diff -= self.prior_mean
            coeffs = params[:, :-1]
            alpha = torch.exp(params[:, -1])
            hyperprior_term = log_hyperprior(alpha)
            prior_term = -torch.sum(torch.square_(prior_diff), dim=-1).mul_(2 * alpha)
            logits = coeffs @ self.data
            likelihood = bernoulli_loglikelihood_logit_v(logits, self.labels)
            # print("alpha:",alpha,"\n\n")
            # print("likely:",likelihood,"\n\n")
            # print("prior:",prior_term,"\n\n")
            # print("hyperprior:",hyperprior_term,"\n\n")
            post = likelihood + prior_term + hyperprior_term
            return post if is_batch else post[0]

        return torch.compile(log_dens) if use_compiled else log_dens


def gaussian_log_dens_factory(
    model: GaussianModel, use_compiled: bool = True
) -> BatchLogDensity:
    def log_dens(pts: BatchPtType) -> BatchType:
        model_eval = model.forward_model(pts)
        obs_error = model_eval.sub_(model.true_obs)
        like_term = torch.square(torch.linalg.norm(obs_error, dim=-1)).mul_(
            model.likelihood_precision
        )
        like_term.mul_(model.likelihood_precision)
        prior_diff = pts
        if model.prior_mean != 0.0:
            prior_diff -= model.prior_mean
        prior_term = torch.square(torch.linalg.norm(prior_diff, dim=-1)).mul_(
            model.prior_precision
        )
        return -0.5 * (prior_term + like_term)

    return torch.compile(log_dens) if use_compiled else log_dens
