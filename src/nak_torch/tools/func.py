from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
import torch
from .types import (
    BatchPtType,
    BatchType,
    DeviceLike,
    BatchTargetEvaluator,
)

BatchTargetEvaluatorT = TypeVar("BatchTargetEvaluatorT", bound=BatchTargetEvaluator)
AlgorithmArgsT = TypeVar("AlgorithmArgsT")
WeightT = TypeVar("WeightT", bound=Optional[BatchType])


class GeneralAdaptiveNAKAlgorithm(
    ABC, Generic[BatchTargetEvaluatorT, WeightT, AlgorithmArgsT]
):
    dim: int
    n_particles: int
    device: Optional[DeviceLike]
    dtype: Optional[torch.dtype]

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike],
        dtype: Optional[torch.dtype],
    ):
        self.dim = dim
        self.n_particles = n_particles
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def initialize(
        self,
        init_particles: BatchPtType,
        target: BatchTargetEvaluatorT,
        target_args: Any,
    ) -> tuple[WeightT, AlgorithmArgsT]:
        pass

    @abstractmethod
    def step(
        self,
        lr: float,
        particles: BatchPtType,
        target: BatchTargetEvaluatorT,
        algorithm_args: AlgorithmArgsT,
        target_args: Any,
    ) -> tuple[BatchPtType, WeightT, AlgorithmArgsT]:
        pass

    @classmethod
    @abstractmethod
    def is_weighted(cls) -> bool:
        pass


class UnweightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[BatchTargetEvaluatorT, None, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return False


class WeightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[BatchTargetEvaluatorT, BatchType, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return True
