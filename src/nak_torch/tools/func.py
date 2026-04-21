from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
import torch
from .types import (
    BatchPtType,
    BatchType,
    DeviceLike,
    BatchDensityEvaluator,
)

BatchDensityEvaluatorT = TypeVar("BatchDensityEvaluatorT", bound=BatchDensityEvaluator)
AlgorithmArgsT = TypeVar("AlgorithmArgsT")
WeightT = TypeVar("WeightT", bound=Optional[BatchType])


class GeneralAdaptiveNAKAlgorithm(
    ABC, Generic[BatchDensityEvaluatorT, WeightT, AlgorithmArgsT]
):
    dim: int
    n_particles: int
    device: Optional[DeviceLike]
    dtype: Optional[torch.dtype]

    @abstractmethod
    def initialize(
        self,
        init_particles: BatchPtType,
        target: BatchDensityEvaluatorT,
        target_args: Any,
    ) -> tuple[WeightT, AlgorithmArgsT]:
        pass

    @abstractmethod
    def step(
        self,
        lr: float,
        particles: BatchPtType,
        target: BatchDensityEvaluatorT,
        algorithm_args: AlgorithmArgsT,
        target_args: Any,
    ) -> tuple[BatchPtType, WeightT, AlgorithmArgsT]:
        pass

    @classmethod
    @abstractmethod
    def is_weighted(cls) -> bool:
        pass


class UnweightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[BatchDensityEvaluatorT, None, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return False


class WeightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[BatchDensityEvaluatorT, BatchType, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return True
