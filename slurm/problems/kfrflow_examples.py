import torch

doughnut_model = torch.compile(lambda pt: torch.linalg.norm(pt, -1))
doughnut_obs = 2.
doughnut_like_var = 0.25**2

butterfly_model = torch.compile(lambda pt: torch.sin(pt[...,1]) + torch.cos(pt[...,0]))
butterfly_obs = -1.
butterfly_like_var = 0.6**2

spaceships_model = torch.compile(lambda pt: torch.sin(pt[...,0] * pt[...,1]) + torch.cos(pt[...,0] * pt[...,1]))
spaceships_obs = -1.
spaceships_like_var = 0.5**2

@torch.compile
def funnel_logpdf(pt: torch.Tensor, var1: float = 9.):
    d = pt.shape[-1]
    x0 = pt[...,0]
    prior_term = torch.sum(pt**2, -1) + torch.log(2*torch.pi*torch.ones(()))
    term_0 = x0**2 / var1 + torch.log(var1*torch.ones(()))
    rest_terms = torch.sum(pt[...,1:]**2,-1) / torch.exp(x0)  + (d-1)*x0
    return -0.5 * (prior_term + term_0 + rest_terms)