# Various functions for illustrations
# twodims: 2D functions
# nns: neural networks
# bayes: posterior distributions for 'realistic' problems
# kde: kernel density estimator, useful to go from discrete to continuous

# Ayoub Belhadji
# 05/12/2025


from .twodims.mixture_of_gaussians import mixture_of_gaussians
from .twodims.himmelblau import himmelblau
from .twodims.banana import banana
from .twodims.ring import ring
from .nns.shallow_nn_twodims import loss_nn_dataset
from .kde.gaussian_kde import gaussian_kde
from .aristoff_bangerth import build_aristoff_bangerth

__all__ = [
    "mixture_of_gaussians",
    "banana",
    "ring",
    "himmelblau",
    "loss_nn_dataset",
    "plot_2D_classification_with_dataset_from_theta",
    "gaussian_kde",
    "build_aristoff_bangerth"
]
