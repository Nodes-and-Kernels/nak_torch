#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of deep ensembles
# Ayoub Belhadji
# 05/12/2025

from nak_torch.tools.func import UnweightedAdaptiveNAKAlgorithm
from nak_torch.tools.types import BatchLogDensityGradEvaluator

__all__ = ["DeepEnsembles"]


class DeepEnsembles(UnweightedAdaptiveNAKAlgorithm[BatchLogDensityGradEvaluator, None]):
    def initialize(self, init_particles, target, target_args):
        return None, None

    def step(self, lr, particles, target, algorithm_args, target_args):
        grad_log_dens_eval = target(particles, target_args)
        new_particles = particles.add(grad_log_dens_eval.mul_(lr))
        return new_particles, None, algorithm_args
