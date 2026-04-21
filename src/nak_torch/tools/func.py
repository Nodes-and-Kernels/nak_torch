from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar
import torch
from .types import (
    BatchPtType,
    BatchType,
    DeviceLike,
    BatchDensityEvaluator,
)

BatchDensityEvaluatorT = TypeVar("BatchDensityEvaluatorT", bound=BatchDensityEvaluator)
AlgorithmArgsT = TypeVar("AlgorithmArgsT")


class AdaptiveNAKAlgorithm(ABC, Generic[BatchDensityEvaluatorT, AlgorithmArgsT]):
    dim: int
    n_particles: int
    device: Optional[DeviceLike]
    dtype: Optional[torch.dtype]

    @abstractmethod
    def __call__(
        self,
        lr: float,
        target: BatchDensityEvaluatorT,
        points: BatchPtType,
        algorithm_args: AlgorithmArgsT,
        target_args,
    ) -> BatchPtType:
        pass

    @abstractmethod
    def update(self, particles: BatchPtType) -> AlgorithmArgsT:
        pass

    def get_weights(self, points: BatchPtType, target_args) -> BatchType:
        N_ens = points.shape[0]
        return torch.ones(N_ens, dtype=points.dtype, device=points.device) / N_ens


class NAKAlgorithm(AdaptiveNAKAlgorithm[BatchDensityEvaluatorT, None]):
    def update(self, particles: BatchPtType) -> None:
        return None


class WeightedAdaptiveNAKAlgorithm(
    AdaptiveNAKAlgorithm[BatchDensityEvaluatorT, AlgorithmArgsT]
):
    @abstractmethod
    def get_weights(self, points: BatchPtType, target_args) -> BatchType:
        pass


class WeightedNAKAlgorithm(NAKAlgorithm[BatchDensityEvaluatorT]):
    @abstractmethod
    def get_weights(self, points: BatchPtType, target_args) -> BatchType:
        pass
