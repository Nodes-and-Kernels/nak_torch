import numpy as np
import torch
from torch import Tensor

class GMM:
    weights: Tensor
    shape: tuple[int,int]
    means: Tensor
    covs: Tensor
    chols: Tensor
    precs: Tensor
    log_constants: Tensor
    def __init__(self, weights, means, covs):
        self.weights = weights
        self.shape = means.shape
        self.means = means
        self.covs = covs
        self.chols = torch.linalg.cholesky(covs)
        self.precs = torch.linalg.inv(covs)
        logdets = torch.logdet(self.precs)
        self.log_constants = weights.log() + 0.5 * (logdets - np.log(2*np.pi))

    def sample(self, N: int, rng = None):
        if rng is None:
            rng = torch.default_generator
        which_modes = torch.multinomial(self.weights, N, True, generator=rng)
        base_rand = torch.randn((N,self.shape[1]), generator=rng)
        scale_rand = torch.einsum("Nij,Nj->Ni", self.chols[which_modes], base_rand)
        return scale_rand + self.means[which_modes]

    def to(self, *args, **kwargs):
        return GMM(self.weights.to(*args, **kwargs), self.means.to(*args, **kwargs), self.covs.to(*args, **kwargs))


def log_density(pt, gmm: GMM):
    pt = pt.reshape(-1, gmm.shape[1])
    diff = pt[:,None,:] - gmm.means[None,:,:]
    exponent = -0.5 * torch.einsum("MKi,Kij,MKj->MK", diff, gmm.precs, diff)
    return torch.logsumexp(exponent + gmm.log_constants[None,:], -1).squeeze()

# ══════════════════════════════════════════════════════════════════════════════
# Closed-form MMD to the target GMM (RBF kernel) — verbatim from the source.
#
# MSIP-Fredholm returns *signed* particle weights. weights=None scores the
# uniform particle measure (comparable to the other examples/gmm scripts);
# passing per-step weights scores its actual signed empirical measure.
# ══════════════════════════════════════════════════════════════════════════════
def gmm_rbf_expectations(particles, gmm: GMM, bandwidth, p_wts=None):
    """Return (E_{y,y'~pi}[k], E_{x~mu, y~pi}[k]) for the RBF kernel k.

    p_wts: optional particle weights for mu (defaults to uniform 1/N).
    """
    sigma_sq = bandwidth**2
    K, D = gmm.shape

    eye = torch.eye(D, device=particles.device, dtype=particles.dtype)
    log_C = 0.5 * D * np.log(2 * np.pi * sigma_sq)

    # E_{y,y'~pi}[k(y,y')] — analytic, independent of the particles.
    Epp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for j in range(K):
        for ell in range(K):
            cov_jl = gmm.covs[j] + gmm.covs[ell] + sigma_sq * eye
            diff_jl = gmm.means[j] - gmm.means[ell]
            log_k = -0.5 * (
                diff_jl @ torch.linalg.solve(cov_jl, diff_jl)
                + torch.logdet(2 * torch.pi * cov_jl)
            ) + log_C
            Epp = Epp + gmm.weights[j] * gmm.weights[ell] * log_k.exp()

    # E_{x~mu, y~pi}[k(x,y)] — mixes particles with the target.
    smoothed_covs = gmm.covs + sigma_sq * eye.unsqueeze(0)
    Exp = torch.tensor(0.0, device=particles.device, dtype=particles.dtype)
    for k in range(K):
        diff_nk = particles - gmm.means[k].unsqueeze(0)
        log_k = -0.5 * (
            torch.einsum(
                "ni,ij,nj->n", diff_nk, torch.linalg.inv(smoothed_covs[k]), diff_nk
            )
            + torch.logdet(2 * torch.pi * smoothed_covs[k])
        ) + log_C
        contrib = log_k.exp()
        contrib = (contrib * p_wts).sum() if p_wts is not None else contrib.mean()
        Exp = Exp + gmm.weights[k] * contrib
    return Epp, Exp


def mmd(particles, gmm: GMM, bandwidth, p_wts=None)->float:
    """sqrt(MMD^2) between the particle measure and the target GMM.

    weights=None  -> uniform particles (matches the other examples).
    weights given -> uses that (possibly signed) empirical measure; normalized
                     to sum to 1 first.
    """
    if p_wts is not None:
        p_wts /= p_wts.sum()

    Epp, Exp = gmm_rbf_expectations(
        particles, gmm, bandwidth, p_wts=p_wts
    )
    Kmat = torch.exp(-torch.cdist(particles, particles).pow(2) / (2 * bandwidth**2))
    if p_wts is not None:
        Kxx = p_wts @ Kmat @ p_wts
    else:
        Kxx = Kmat.mean()
    return (Kxx + Epp - 2 * Exp).sqrt().item()
