from typing import Any, Callable, Optional, Protocol, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import Tensor
from torch.utils import data as torch_data
from jaxtyping import Float, Bool
from typing import Self

DeviceLike = str | torch.device | int

BatchType = Float[Tensor, "batch"]
PtType = Float[Tensor, " d"]
CovType = Float[Tensor, "d d"]
BatchPtType = Float[Tensor, "batch d"]
DatasetType = Bool[Tensor, "d samples"]
LabelsType = Bool[Tensor, " batch"]
QuadrulePtType = Float[Tensor, "quad d"]
QuadruleWtType = Float[Tensor, "quad"]
BatchQuadrulePtType = Float[Tensor, "batch quad d"]
BatchQuadruleWtType = Float[Tensor, "batch quad"]
KernelMatrixType = Float[Tensor, "batch batch"]
GradKernelMatrixType = Float[Tensor, "batch batch d"]


DensityGradValOutput = tuple[BatchPtType, BatchType]
MSIPEstimatorOutput = tuple[BatchType, BatchPtType]

KernelFunction = Callable[[PtType, PtType, float], Float]
BatchKernelGradValFunction = Callable[
    [BatchPtType, BatchPtType, Any], tuple[BatchPtType, Float]
]

EvaluatorOutputT = TypeVar("EvaluatorOutputT")


class NAKTarget(ABC, Generic[EvaluatorOutputT]): ...


class BatchTargetEvaluator(NAKTarget[EvaluatorOutputT]):
    @abstractmethod
    def __call__(self, particles: BatchPtType, target_args) -> EvaluatorOutputT:
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


class BatchLogDensityEvaluator(BatchTargetEvaluator[BatchType]):
    log_density: BatchLogDensity

    def __init__(self, log_density: LogDensity | BatchLogDensity, is_batched: bool):
        if not is_batched:
            log_density = torch.vmap(log_density, in_dims=(0, None))
        self.log_density = log_density

    def __call__(self, pts, target_args):
        return self.log_density(pts, target_args)


class BatchLogDensityGradValEvaluator(
    BatchTargetEvaluator[tuple[BatchPtType, BatchType]]
):
    grad_val_log_density: BatchLogDensityGradVal

    def __init__(self, log_density: LogDensity | BatchLogDensity, is_batched: bool):
        if is_batched:

            def aux_lam(x: BatchPtType, p: Any) -> tuple[Float, BatchType]:
                log_pi_x = log_density(x, p)
                return log_pi_x.sum(), log_pi_x

            self.grad_val_log_density = torch.func.grad(aux_lam, has_aux=True)
        else:
            grad_val_log_density = torch.func.grad_and_value(log_density)
            self.grad_val_log_density = torch.vmap(
                grad_val_log_density, in_dims=(0, None)
            )
        self.log_density = log_density

    def __call__(self, pts, target_args):
        return self.grad_val_log_density(pts, target_args)


class BatchLogDensityGradEvaluator(BatchTargetEvaluator[BatchPtType]):
    grad_log_density: BatchGradLogDensity

    def __init__(
        self,
        log_density_or_grad: LogDensity
        | BatchLogDensity
        | GradLogDensity
        | BatchGradLogDensity,
        is_grad: bool,
        is_batched: bool,
    ):
        if is_batched:
            if is_grad:
                self.grad_log_density = log_density_or_grad
            else:
                self.grad_log_density = torch.func.grad(
                    lambda x, args: log_density_or_grad(x, args).sum()
                )
        else:
            if not is_grad:
                log_density_or_grad = torch.func.grad(log_density_or_grad)
            self.grad_log_density = torch.vmap(log_density_or_grad, in_dims=(0, None))

    def __call__(self, pts, target_args):
        return self.grad_log_density(pts, target_args)


class AbstractModel(NAKTarget):
    @abstractmethod
    def to_log_dens(self: Self, use_compiled: bool = True) -> BatchLogDensity:
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
            like_sq_norm = obs_error.square().sum(dim=-1)
            like_term = like_sq_norm.mul_(self.likelihood_precision)
            prior_diff = pts.clone()
            if self.prior_mean != 0.0:
                prior_diff.sub_(self.prior_mean)
            prior_sq_norm = prior_diff.square().sum(dim=-1)
            prior_term = prior_sq_norm.mul_(self.prior_precision)
            return -0.5 * (prior_term + like_term)

        return torch.compile(log_dens) if use_compiled else log_dens


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
    train_data: Float | Float[Tensor, "dim labels"]
    test_data: Optional[Float | Float[Tensor, "dim labels"]]
    train_labels: Float | Float[Tensor, " labels"]
    test_labels: Optional[Float | Float[Tensor, " labels"]]
    use_mean_reduction: bool
    hyperprior: torch.distributions.Gamma

    def __init__(
        self,
        data_or_fname: Float[Tensor, "labels dim-1"] | str,
        labels: Optional[Float[Tensor, " labels"]],
        prior_mean: float | Float[Tensor, " dim"] | None = None,
        dtype=None,
        device=None,
        hyperprior_a=1.0,
        hyperprior_b=0.1,
        train_proportion=1.0,
        reduction="mean",
    ):
        data: torch.Tensor
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device

        def as_tensor(t):
            return torch.as_tensor(t, dtype=dtype, device=device)

        match reduction:
            case "mean":
                self.use_mean_reduction = True
            case "sum":
                self.use_mean_reduction = False
            case _:
                raise ValueError(
                    f"Expected reduction to be sum or mean, got {reduction}"
                )
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
        N_pts = data.shape[0]
        if labels is None or labels.shape[0] != N_pts:
            raise ValueError("Unexpected type or size of argument `labels`.")
        constant = as_tensor(torch.ones(N_pts))
        data = torch.column_stack((constant, data))
        if train_proportion >= 1.0:
            self.train_data, self.test_data = data, None
            self.train_labels, self.test_labels = labels, None
        else:
            ridx = torch.randperm(N_pts)
            num_train = int(np.floor(N_pts * train_proportion))
            self.train_data = data[ridx[:num_train]]
            self.train_labels = labels[ridx[:num_train]]
            self.test_data = data[ridx[num_train:]]
            self.test_labels = labels[ridx[num_train:]]
        self.dim = data.shape[1] + 1
        self.prior_mean = prior_mean
        self.hyperprior = torch.distributions.Gamma(
            as_tensor(hyperprior_a), as_tensor(hyperprior_b)
        )

    def to_log_dens(self, use_compiled: bool = True):
        log_hyperprior = self.hyperprior.log_prob

        def log_dens(
            params: BatchPtType,
            data_labels: Optional[tuple[BatchPtType, LabelsType]] = None,
        ) -> BatchType:
            total_N = self.train_data.shape[0]
            if data_labels is None:
                data, labels = self.train_data, self.train_labels
            else:
                data, labels = data_labels
            is_batch = params.ndim == 2
            if not is_batch:
                params = params.unsqueeze(0)
            if params.shape[1] != self.dim:
                raise ValueError(
                    f"Got params.shape[1] = {params.shape[1]}, expected {self.dim}"
                )
            coeffs = params[:, :-1]
            log_precision = params[:, -1]
            prior_diff = coeffs.clone()
            if self.prior_mean is not None:
                prior_diff -= self.prior_mean
            precision = torch.exp(log_precision)
            # Correct for change-of-variables precision using chain rule:
            # exp(log_precision) -> log(d_log_precision) = log(d_precision) + log_precision
            hyperprior_term = log_hyperprior(precision) + log_precision
            prior_term = prior_diff.square().sum(dim=-1).mul_(-0.5 * precision)
            # log-normalization constant of prior w.r.t. alpha = precision
            num_coeffs = coeffs.shape[1]
            prior_term = prior_term.add_((0.5 * num_coeffs) * log_precision)
            logits = coeffs @ data.T
            likelihood = bernoulli_loglikelihood_logit_v(logits, labels)
            if self.use_mean_reduction:
                likelihood *= total_N / labels.numel()
            post = likelihood + prior_term + hyperprior_term
            return post if is_batch else post[0]

        return torch.compile(log_dens) if use_compiled else log_dens

    def get_data_loader(
        self,
        use_test_data: bool,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        *data_loader_args,
        **data_loader_kwargs,
    ):
        data: torch_data.TensorDataset
        if use_test_data:
            if self.test_data is None or self.test_labels is None:
                raise ValueError("Cannot use test data as None")
            data = torch_data.TensorDataset(self.test_data, self.test_labels)
        else:
            data = torch_data.TensorDataset(self.train_data, self.train_labels)
        return torch_data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *data_loader_args,
            **data_loader_kwargs,
        )
