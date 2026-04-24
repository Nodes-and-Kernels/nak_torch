from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
import warnings
import torch
from .types import (
    BatchPtType,
    BatchType,
    DeviceLike,
    NAKTarget,
)

NAKTargetT = TypeVar("NAKTargetT", bound=NAKTarget)
AlgorithmArgsT = TypeVar("AlgorithmArgsT")
WeightT = TypeVar("WeightT", bound=Optional[BatchType])


class GeneralAdaptiveNAKAlgorithm(ABC, Generic[NAKTargetT, WeightT, AlgorithmArgsT]):
    dim: int
    n_particles: int
    device: Optional[DeviceLike]
    dtype: Optional[torch.dtype]
    verbose: bool

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike],
        dtype: Optional[torch.dtype],
        verbose: bool = True,
        **kwargs,
    ):
        self.dim = dim
        self.n_particles = n_particles
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        if verbose and len(kwargs) > 0:
            warnings.warn(f"Unused kwargs:\n{kwargs}")

    @abstractmethod
    def initialize(
        self,
        init_particles: BatchPtType,
        target: NAKTargetT,
        target_args: Any,
    ) -> tuple[WeightT, AlgorithmArgsT]:
        pass

    @abstractmethod
    def step(
        self,
        lr: float,
        particles: BatchPtType,
        target: NAKTargetT,
        algorithm_args: AlgorithmArgsT,
        target_args: Any,
    ) -> tuple[BatchPtType, WeightT, AlgorithmArgsT]:
        pass

    @classmethod
    @abstractmethod
    def is_weighted(cls) -> bool:
        pass


class UnweightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[NAKTargetT, None, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return False


class WeightedAdaptiveNAKAlgorithm(
    GeneralAdaptiveNAKAlgorithm[NAKTargetT, BatchType, AlgorithmArgsT]
):
    @classmethod
    def is_weighted(cls) -> bool:
        return True
