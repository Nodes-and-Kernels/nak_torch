# -----------------------------------------------------------------------------
#
# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright (C) 2019 by Wolfgang Bangerth
#
# This file is redistributed code with changes from the deal.II code gallery.
#
# -----------------------------------------------------------------------------
#
# Author: Wolfgang Bangerth, Colorado State University, 2019.
# Author: NAK torch authors, 2026
#

from functools import partial

import torch
from torch import Tensor
from typing import Optional
from jaxtyping import Float
from nak_torch import GaussianModel
from nak_torch.tools.types import BatchForwardModel, BatchPtType
if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

theta_true = torch.tensor([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 10., 10., 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
])

z_hat_true = torch.tensor([
    0.06076511762259369, 0.09601910120848481,
    0.1238852517838584,  0.1495184117375201,
    0.1841596127549784,  0.2174525028261122,
    0.2250996160898698,  0.2197954769002993,
    0.2074695698370926,  0.1889996477663016,
    0.1632722532153726,  0.1276782480038186,
    0.07711845915789312, 0.09601910120848552,
    0.2000589533367983,  0.3385592591951766,
    0.3934300024647806,  0.4040223892461541,
    0.4122329537843092,  0.4100480091545554,
    0.3949151637189968,  0.3697873264791232,
    0.33401826235924,    0.2850397806663382,
    0.2184260032478671,  0.1271121156350957,
    0.1238852517838611,  0.3385592591951819,
    0.7119285162766475,  0.8175712861756428,
    0.6836254116578105,  0.5779452419831157,
    0.5555615956136897,  0.5285181561736719,
    0.491439702849224,   0.4409367494853282,
    0.3730060082060772,  0.2821694983395214,
    0.1610176733857739,  0.1495184117375257,
    0.3934300024647929,  0.8175712861756562,
    0.9439154625527653,  0.8015904115095128,
    0.6859683749254024,  0.6561235366960599,
    0.6213197201867315,  0.5753611315000049,
    0.5140091754526823,  0.4325325506354165,
    0.3248315148915482,  0.1834600412730086,
    0.1841596127549917,  0.4040223892461832,
    0.6836254116578439,  0.8015904115095396,
    0.7870119561144977,  0.7373108331395808,
    0.7116558878070463,  0.6745179049094283,
    0.6235300574156917,  0.5559332704045935,
    0.4670304994474178,  0.3499809143811,
    0.19688263746294,    0.2174525028261253,
    0.4122329537843404,  0.5779452419831566,
    0.6859683749254372,  0.7373108331396063,
    0.7458811983178246,  0.7278968022406559,
    0.6904793535357751,  0.6369176452710288,
    0.5677443693743215,  0.4784738764865867,
    0.3602190632823262,  0.2031792054737325,
    0.2250996160898818,  0.4100480091545787,
    0.5555615956137137,  0.6561235366960938,
    0.7116558878070715,  0.727896802240657,
    0.7121928678670187,  0.6712187391428729,
    0.6139157775591492,  0.5478251665295381,
    0.4677122687599031,  0.3587654911000848,
    0.2050734291675918,  0.2197954769003094,
    0.3949151637190157,  0.5285181561736911,
    0.6213197201867471,  0.6745179049094407,
    0.690479353535786,   0.6712187391428787,
    0.6178408289359514,  0.5453605027237883,
    0.489575966490909,   0.4341716881061278,
    0.3534389974779456,  0.2083227496961347,
    0.207469569837099,   0.3697873264791366,
    0.4914397028492412,  0.5753611315000203,
    0.6235300574157017,  0.6369176452710497,
    0.6139157775591579,  0.5453605027237935,
    0.4336604929612851,  0.4109641743019312,
    0.3881864790111245,  0.3642640090182592,
    0.2179599909280145,  0.1889996477663011,
    0.3340182623592461,  0.4409367494853381,
    0.5140091754526943,  0.5559332704045969,
    0.5677443693743304,  0.5478251665295453,
    0.4895759664908982,  0.4109641743019171,
    0.395727260284338,   0.3778949322004734,
    0.3596268271857124,  0.2191250268948948,
    0.1632722532153683,  0.2850397806663325,
    0.373006008206081,   0.4325325506354207,
    0.4670304994474315,  0.4784738764866023,
    0.4677122687599041,  0.4341716881061055,
    0.388186479011099,   0.3778949322004602,
    0.3633362567187364,  0.3464457261905399,
    0.2096362321365655,  0.1276782480038148,
    0.2184260032478634,  0.2821694983395252,
    0.3248315148915535,  0.3499809143811097,
    0.3602190632823333,  0.3587654911000799,
    0.3534389974779268,  0.3642640090182283,
    0.35962682718569,    0.3464457261905295,
    0.3260728953424643,  0.180670595355394,
    0.07711845915789244, 0.1271121156350963,
    0.1610176733857757,  0.1834600412730144,
    0.1968826374629443,  0.2031792054737354,
    0.2050734291675885,  0.2083227496961245,
    0.2179599909279998,  0.2191250268948822,
    0.2096362321365551,  0.1806705953553887,
    0.1067965550010013
], dtype=torch.float64)

z_hat_noisy = torch.tensor([
    0.038552099180378,  0.045816483525318,
    0.081592383259465,  0.087475902315745,
    0.199035729421209,  0.237941119489265,
    0.245625512426379,  0.228441511573012,
    0.192291421229205,  0.191451034701013,
    0.153454842617822,  0.119236755151769,
    -0.003220511807785, 0.010973017426285,
    0.193965038755027,  0.384597632561609,
    0.442138416366311,  0.343303595549975,
    0.430288220205305,  0.40991009383553,
    0.39076937430488,   0.408021664044339,
    0.311542533481501,  0.298102825218499,
    0.070986777897544,  0.162806618415527,
    0.180372949412782,  0.35093032775436,
    0.679014086564853,  0.855075076483386,
    0.630617027411117,  0.594983531012196,
    0.579115434478466,  0.52262832039197,
    0.45314767654776,   0.428033530273219,
    0.44251939662937,   0.282630825757986,
    0.148014658259268,  0.112442970157067,
    0.435339458625886,  0.765142734856627,
    0.925338560676569,  0.791992535796009,
    0.650100323418336,  0.600620010142976,
    0.649538488553252,  0.568964305296719,
    0.559278486371505,  0.429491280648342,
    0.241888889228535,  0.194564172555098,
    0.142083696304824,  0.452684945371874,
    0.666074664880897,  0.823534343368938,
    0.789135420881139,  0.709000069097871,
    0.683542563105279,  0.694946288447186,
    0.671473930751091,  0.448435155854656,
    0.459393239079529,  0.297816937602741,
    0.143700347985899,  0.169538768296654,
    0.376101527132257,  0.594909314496177,
    0.692831824259869,  0.696165899428013,
    0.824204839051023,  0.713568916471968,
    0.665775045984689,  0.585057421192582,
    0.53238325698633,   0.424933255372099,
    0.348457701085593,  0.191225627478046,
    0.331904057716994,  0.400868289067284,
    0.554683400728588,  0.637404680551052,
    0.611461257262823,  0.778471101668002,
    0.749609970080417,  0.671970715208512,
    0.653289712318449,  0.546698832213113,
    0.425423474347357,  0.317667608207469,
    0.237699981471772,  0.247344535395687,
    0.41910073890388,   0.642188425337278,
    0.587244707291769,  0.712856859806251,
    0.732698302575663,  0.68985830648619,
    0.582051029869533,  0.580400352056727,
    0.465513933407401,  0.447153103893242,
    0.493907282955395,  0.179341681613428,
    0.247557683839442,  0.353446454783813,
    0.407362153648493,  0.5783575375043,
    0.561784598253918,  0.581109383541697,
    0.631322293306865,  0.530347919084227,
    0.511014241817423,  0.38176330708819,
    0.400875758936058,  0.354488526553193,
    0.165037466036415,  0.203917937263943,
    0.347283241928518,  0.434746754511574,
    0.531679386029285,  0.612104882401232,
    0.521162956777221,  0.477515683848052,
    0.481028068732176,  0.394277451722145,
    0.439164199716157,  0.394606284291556,
    0.377074532641213,  0.215265427841687,
    0.115320458528513,  0.235533234924869,
    0.357628577926302,  0.455619894927127,
    0.445130691277276,  0.550013261800976,
    0.509016896462794,  0.396931633056656,
    0.349149684056181,  0.361147321057958,
    0.374918770168706,  0.295962046051228,
    0.297892842409811,  0.144389758610725,
    0.224471309274922,  0.348968334751183,
    0.299385447107182,  0.422371728734795,
    0.383600811392348,  0.322907774351887,
    0.323922749871539,  0.352698232706256,
    0.296604187718578,  0.402010403122009,
    0.369674733918912,  0.225267911730855,
    0.141935954280098,  0.101402467757723,
    0.194181814632711,  0.134022566736322,
    0.133909855334354,  0.263242690971926,
    0.150905738905573,  0.237657643431622,
    0.204195242165641,  0.208483186059569,
    0.252630970410341,  0.169066197495019,
    0.163995346152564
], dtype=torch.float64)

def heaviside(x: Tensor):
    return (x >= 0).type_as(x)

def S(x, y, h):
    return heaviside(x)*heaviside(y) * (1-heaviside(x-h))*(1-heaviside(y-h))

# Define tent function on the domain [0,2h]x[0,2h]
def phi(x, y, h):
    return (
        (x+h)*(y+h)*S(x+h, y+h, h) + (h-x)*(h-y)*S(x, y, h)
        + (x+h)*(h-y)*S(x+h, y, h) + (h-x)*(y+h)*S(x, y+h, h)
    )/h**2

# Define conversion function for dof's from 2D to scalar label, and
# its inverse
def ij_to_dof_index(i, j, N):
    return (N + 1)*j+i

def inv_ij_to_dof_index(k, N):
    Np1 = N + 1
    return [k-Np1*(k // Np1), (k // Np1)]

def build_forward_solver_args(N, N_obs, device=None, dtype: Optional[torch.dtype]=torch.float32):
    h = 1 / N
    Np1 = N+1
    # Construct measurement matrix, M, for measurements
    xs = torch.arange(1, N_obs+1, dtype=dtype, device=device) / (N_obs + 1)  # measurement points

    XS_1, XS_2 = torch.meshgrid(xs, xs, indexing='ij')

    M = torch.empty((Np1**2, N_obs**2), dtype=dtype, device=device)
    for k in range(Np1**2):
        c = inv_ij_to_dof_index(k, N)
        slice = phi(XS_1 - h*c[0], XS_2 - h*c[1], h)
        M[k] = slice.flatten()
    M.transpose_(1, 0)
    # Construct local overlap matrix, A_loc, and identity matrix Id
    A_loc = torch.tensor([
        [2./3,  -1./6,  -1./3,  -1./6],
        [-1./6,  2./3,  -1./6,  -1./3],
        [-1./3, -1./6,   2./3,  -1./6],
        [-1./6, -1./3,  -1./6,   2./3]
    ], dtype=dtype, device=device)

    # Locate boundary labels
    boundaries = torch.concat((
        ij_to_dof_index(torch.arange(Np1, device=device), 0, N),
        ij_to_dof_index(torch.arange(Np1, device=device), N, N),
        ij_to_dof_index(0, torch.arange(1,N, device=device), N),
        ij_to_dof_index(N, torch.arange(1,N, device=device), N),
    ))
    # boundaries = torch.tensor(boundaries)
    # Define RHS of FEM linear system, AU = b
    b = torch.ones(Np1**2, dtype=dtype, device=device)*10*h**2
    # enforce boundary conditions on b
    b[boundaries] = 0
    return M, boundaries, A_loc, b

###########################################################################
###################### forward solver function ############################
###########################################################################
def forward_solver(
        theta: BatchPtType, # (N_batch, 64)
        N: int,
        M: Float[Tensor, "obs grid"], # (N_obs, N)
        boundaries: Float[Tensor, " 4*grid"], # (4*N,),
        A_loc: Float[Tensor, "p p"], # (4, 4)
        b: Float[Tensor, " (grid+1)**2"], # ((N+1)**2, )
) -> Float[Tensor, "batch obs"]: # (N+1, )
    """
    Solve Poisson PDE for Aristoff-Bangerth example.

    :param theta: positive-valued parameters
    :type theta: Float[Array, "64"]
    :param N: Mesh size
    :type N: int
    :param M: Observation operator
    :type M: Float[Array, "N_obs N"]
    :param boundaries: Boundary values
    :type boundaries: Float[Array, "*"]
    :param A_loc: Local FEM matrix
    :type A_loc: Float[Array, "4 4"]
    :param b: PDE right-hand side
    :type b: Float[Array, "(N+1)**2"]
    :return: Evaluation of PDE on mesh
    :rtype: Array
    """
    # Initialize matrix A for FEM linear solve, AU = b
    if theta.ndim != 2:
        raise ValueError(f"Expected theta.ndim = 2, got {theta.ndim}")
    if theta.shape[1] != 64:
        raise ValueError(f"Expected theta.shape[1] = 64, got {theta.shape[1]}")
    N_batch = theta.shape[0]
    Np1 = N + 1
    num_patch = 8
    frac = num_patch / N
    # Build A by summing over contribution from each cell
    patch_vals = theta.reshape(N_batch, 64, 1, 1) * A_loc.reshape(1, 1, 4, 4)
    device = theta.device
    def get_patch_idx(idx):
        i, j = idx // N, idx % N
        patch_i, patch_j = torch.floor(i * frac), torch.floor(j * frac)
        return i, j, torch.as_tensor(patch_i + patch_j * num_patch, dtype=torch.int32, device=device)
    i_all, j_all, patch_idxs = torch.vmap(get_patch_idx)(torch.arange(N**2, device=device))
    A_locs = patch_vals[:, patch_idxs]
    A_idxs = torch.stack((
        ij_to_dof_index(i_all, j_all, N),
        ij_to_dof_index(i_all, j_all + 1, N),
        ij_to_dof_index(i_all + 1, j_all + 1, N),
        ij_to_dof_index(i_all + 1, j_all, N)
    )).repeat(4,1).reshape(4,4,-1).permute((2,0,1))

    A_rows = A_idxs
    A_cols = A_idxs.permute((0,2,1))
    A_dens = torch.zeros((N_batch, Np1**2, Np1**2), device=theta.device, dtype=theta.dtype)
    # Unroll loop
    A_dens[:,A_rows[:,0,0],A_cols[:,0,0]] += A_locs[:,:,0,0]
    A_dens[:,A_rows[:,0,1],A_cols[:,0,1]] += A_locs[:,:,0,1]
    A_dens[:,A_rows[:,0,2],A_cols[:,0,2]] += A_locs[:,:,0,2]
    A_dens[:,A_rows[:,0,3],A_cols[:,0,3]] += A_locs[:,:,0,3]
    A_dens[:,A_rows[:,1,0],A_cols[:,1,0]] += A_locs[:,:,1,0]
    A_dens[:,A_rows[:,1,1],A_cols[:,1,1]] += A_locs[:,:,1,1]
    A_dens[:,A_rows[:,1,2],A_cols[:,1,2]] += A_locs[:,:,1,2]
    A_dens[:,A_rows[:,1,3],A_cols[:,1,3]] += A_locs[:,:,1,3]
    A_dens[:,A_rows[:,2,0],A_cols[:,2,0]] += A_locs[:,:,2,0]
    A_dens[:,A_rows[:,2,1],A_cols[:,2,1]] += A_locs[:,:,2,1]
    A_dens[:,A_rows[:,2,2],A_cols[:,2,2]] += A_locs[:,:,2,2]
    A_dens[:,A_rows[:,2,3],A_cols[:,2,3]] += A_locs[:,:,2,3]
    A_dens[:,A_rows[:,3,0],A_cols[:,3,0]] += A_locs[:,:,3,0]
    A_dens[:,A_rows[:,3,1],A_cols[:,3,1]] += A_locs[:,:,3,1]
    A_dens[:,A_rows[:,3,2],A_cols[:,3,2]] += A_locs[:,:,3,2]
    A_dens[:,A_rows[:,3,3],A_cols[:,3,3]] += A_locs[:,:,3,3]

    A_dens[:,boundaries, :] = 0.
    A_dens[:,:,boundaries] = 0.
    A_dens[:,boundaries, boundaries] = 1.
    # Solve linear equation for coefficients, U, and then
    # get the Z vector by multiplying by the measurement matrix
    return torch.linalg.solve(A_dens, b.repeat(N_batch,1))

def log_likelihood(log_theta: Tensor, N, z_hat: Tensor, sig_lik_sq: float, H_obs: Tensor, *solve_args):
    theta = log_theta.exp()
    out = forward_solver(theta, N, H_obs, *solve_args)
    diff = (out @ H_obs.T) - z_hat
    norm_sq = diff.square().sum(-1)
    return -norm_sq/(2*sig_lik_sq)

def log_prior(log_theta: Tensor, sig_pr_sq: float):
    norm_sq = log_theta.square().sum(-1)
    return -norm_sq / (2 * sig_pr_sq)

def build_forward_solver(N: int, H_obs: Tensor, *solve_args) -> BatchForwardModel:
    def prefill_forward_solver(log_theta: Tensor, _ = None):
        return forward_solver(log_theta.exp(), N, H_obs, *solve_args) @ H_obs.T
    return prefill_forward_solver

def build_aristoff_bangerth(
        N: int = 32,
        N_obs: int = 13,
        sig_lik: float = 0.05,
        sig_pr: float = 2.0,
        z_hat: Optional[Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        use_compiled: bool = True,
        device: Optional[torch.device] = None
):
    if z_hat is None:
        z_hat = z_hat_noisy
    if z_hat.ndim != 1 or z_hat.shape[0] != N_obs**2:
        raise ValueError("Unexpected number of dimensions for z hat")
    if dtype is None:
        dtype = z_hat.dtype
    if device is None:
        device = torch.get_default_device()
    z_hat = torch.as_tensor(z_hat, dtype=dtype, device=device)
    sig_lik_sq = sig_lik**2
    sig_pr_sq = sig_pr**2
    solve_args = build_forward_solver_args(N, N_obs, dtype=dtype, device=device)
    forward_solver_func = build_forward_solver(N, *solve_args)
    if use_compiled:
        forward_solver_func = torch.compile(forward_solver_func)
    model = GaussianModel(forward_solver_func, 1/sig_lik_sq, 1/sig_pr_sq, z_hat, is_vectorized=True)
    return model


def verify_against_stored_tests(path, z_hat, dtype=torch.float64):
    N, N_obs = 32, 13
    solve_args = build_forward_solver_args(N, N_obs, dtype=dtype)

    def get_file(fname):
        return open("{}/{}.txt".format(path, fname), 'r')

    for i in range(10):
        print("Verifying against data set", i)
        # Read the input vector
        f_ijnput = get_file("input.{}".format(i))
        theta = torch.tensor(np.fromfile(f_ijnput, count=64, sep=" "), dtype=dtype)
        # Then computes both the forward solution and its statistics.
        # This is not efficiently written here (it calls the forward
        # solver twice), but we don't care about efficiency here since
        # we are only computing with ten samples
        sig_lik, sig_pr = 0.05, 2.
        this_log_likelihood = log_likelihood(
            theta.log(), N, z_hat, sig_lik**2, *solve_args
        ).item()
        this_log_prior = log_prior(theta.log(), sig_pr**2).item()

        # Then also read the reference output generated by the C++ program:
        f_output_z = get_file("output.{}.z".format(i))
        f_output_likelihood = get_file("output.{}.loglikelihood".format(i))
        f_output_prior = get_file("output.{}.logprior".format(i))

        reference_z = torch.tensor(np.fromfile(
            f_output_z, count=13**2, sep=" "))
        reference_log_likelihood = float(f_output_likelihood.read())
        reference_log_prior = float(f_output_prior.read())
        norm = torch.linalg.norm
        error_z = norm(z_hat - reference_z) / norm(reference_z)
        error_LL = abs(
            (this_log_likelihood - reference_log_likelihood) /
            reference_log_likelihood
        )
        error_prior = abs(this_log_prior - reference_log_prior)
        if abs(reference_log_prior) > 1e-10:
            error_prior /= abs(reference_log_prior)

        print("  || z-z_ref ||  : ", error_z)
        print("  log likelihood : ",
              "Python value=", this_log_likelihood,
              "(C++ reference value=", reference_log_likelihood,
              ", error=", error_LL,
              ")"
              )
        print("  log prior      : ",
              "Python value=", this_log_prior,
              "(C++ reference value=", reference_log_prior,
              ", error=", error_prior,
              ")\n\n"
              )

if __name__ == '__main__':
    import sys
    import numpy as np
    if len(sys.argv) < 1:
        raise ValueError("Expected path as first argument to this script")
    path = sys.argv[1]
    verify_against_stored_tests(path, z_hat_true)
