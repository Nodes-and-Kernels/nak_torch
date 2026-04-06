from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Callable, Generic, NamedTuple, Optional, TypeVar
import torch
from torch import Tensor
from jaxtyping import Float, Integer, Bool
from enum import IntEnum

# Code adapted from `torchode` library (MIT License for redistribution)
# GitHub: https://github.com/martenlienen/torchode
# BibTex:
# @inproceedings{lienen2022torchode,
#   title = {torchode: A Parallel {ODE} Solver for PyTorch},
#   author = {Marten Lienen and Stephan G{\"u}nnemann},
#   booktitle = {The Symbiosis of Deep Learning and Differential Equations II, NeurIPS},
#   year = {2022},
#   url = {https://openreview.net/forum?id=uiKVKTiUYB0}
# }

DataTensor = Float[Tensor, "batch feature"]
NormTensor = Float[Tensor, "batch"]
TimeTensor = Float[Tensor, "batch"]
StatusTensor = Integer[Tensor, "batch"]
AcceptTensor = Bool[Tensor, "batch"]
CoefficientVector = Float[Tensor, " nodes"]
RungeKuttaMatrix = Float[Tensor, "nodes weights"]
WeightVector = Float[Tensor, " weights"]
WeightMatrix = Float[Tensor, "rows weights"]
NormFunction = Callable[[DataTensor], NormTensor]
DiffFunction = Callable[[TimeTensor, DataTensor, Any], DataTensor]


class status_codes(IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    REACHED_DT_MIN = 2
    REACHED_MAX_STEPS = 3
    INFINITE_NORM = 4


class StepResult(NamedTuple):
    y: DataTensor
    error_estimate: Optional[DataTensor]


class ODETerm(torch.nn.Module):
    vf: DiffFunction

    def __init__(self, diff):
        super().__init__()
        self.vf = diff


class ButcherTableau:
    def __init__(
        self,
        # Coefficients for the evaluation nodes in time
        c: CoefficientVector,
        # Runge-Kutta matrix
        a: RungeKuttaMatrix,
        # Coefficients for the high-order solution estimate
        b: WeightVector,
        # Coefficients for the error estimate
        b_err: WeightVector,
        # Additional additional rows of the b matrix
        b_other: Optional[WeightMatrix] = None,
        fsal: Optional[bool] = None,
        ssal: Optional[bool] = None,
    ):
        self.c = c
        self.a = a
        self.b = b
        self.b_err = b_err
        self.b_other = b_other

        if fsal is None:
            fsal = self.is_fsal()
        self.fsal = fsal
        if ssal is None:
            ssal = self.is_ssal()
        self.ssal = ssal

    @staticmethod
    def from_lists(
        *,
        c: list[float],
        a: list[list[float]],
        b: list[float],
        b_err: Optional[list[float]] = None,
        b_low_order: Optional[list[float]] = None,
        b_other: Optional[list[list[float]]] = None,
    ):
        assert b_err is not None or b_low_order is not None, (
            "You have to provide either the weights for the error approximation"
            " or the weights of an embedded lower-order method"
        )

        n_nodes = len(c)
        n_weights = len(b)
        assert n_nodes == n_weights
        assert len(a) == n_nodes

        # Fill a up into a full square matrix
        a_full = [row + [0.0] * (n_weights - len(row)) for row in a]

        b_coeffs = torch.tensor(b, dtype=torch.float64)
        if b_err is None:
            assert b_low_order is not None
            assert len(b_low_order) == n_weights
            b_low_coeffs = torch.tensor(b_low_order, dtype=torch.float64)
            b_err_coeffs = b_coeffs - b_low_coeffs
        else:
            b_err_coeffs = torch.tensor(b_err, dtype=torch.float64)

        if b_other is None:
            b_other_coeffs = None
        else:
            b_other_coeffs = torch.tensor(b_other, dtype=torch.float64)
            assert b_other_coeffs.ndim == 2
            assert b_other_coeffs.shape[1] == n_weights

        return ButcherTableau(
            c=torch.tensor(c, dtype=torch.float64),
            a=torch.tensor(a_full, dtype=torch.float64),
            b=b_coeffs,
            b_err=b_err_coeffs,
            b_other=b_other_coeffs,
        )

    def to(
        self, device: torch.device, time_dtype: torch.dtype, data_dtype: torch.dtype
    ) -> "ButcherTableau":
        b_other = self.b_other
        if b_other is not None:
            b_other = b_other.to(device, data_dtype)
        return ButcherTableau(
            c=self.c.to(device, time_dtype),
            a=self.a.to(device, data_dtype),
            b=self.b.to(device, data_dtype),
            b_err=self.b_err.to(device, data_dtype),
            b_other=b_other,
            fsal=self.fsal,
            ssal=self.ssal,
        )

    @property
    def n_stages(self):
        return self.c.shape[0]

    def is_fsal(self) -> bool:
        """Is `f(y0)` equal to `f(y1)` from the previous step?

        If that is the case, we can reuse the result from the previous step.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all()
        first_node_is_t0 = self.c[0] == 0.0
        last_node_is_t1 = self.c[-1] == 1.0
        first_stage_explicit = self.a[0, 0] == 0.0
        ret = (
            is_lower_triangular
            & first_node_is_t0
            & last_node_is_t1
            & first_stage_explicit
        )
        return bool(ret.to(dtype=torch.bool).item())

    def is_ssal(self) -> bool:
        """Is the solution equal to the last stage result?

        If that is the case, we can avoid the final computation of the solution and
        return the last stage result instead.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all()
        last_node_is_t1 = self.c[-1] == 1.0
        last_stage_explicit = self.a[-1, -1] == 0.0
        ret = is_lower_triangular & last_node_is_t1 & last_stage_explicit
        return bool(ret.item())


class StepMethodState:
    pass


StepMethodStateBound = TypeVar("StepMethodStateBound", bound=StepMethodState)


class StepMethod(Generic[StepMethodStateBound], ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init(
        self,
        y0: DataTensor,
        term: Optional[ODETerm],
        f0: Optional[DataTensor],
        *,
        args: Any,
        t0: float = 0.0,
    ) -> StepMethodStateBound:
        pass

    @abstractmethod
    def merge_states(
        self,
        accept: AcceptTensor,
        current: StepMethodStateBound,
        previous: StepMethodStateBound,
    ) -> StepMethodStateBound:
        pass

    @abstractmethod
    def step(
        self,
        term: Optional[ODETerm],
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: StepMethodStateBound,
        *,
        args: Any,
    ) -> tuple[StepResult, StepMethodStateBound, Optional[StatusTensor]]:
        pass


@dataclass
class ERKState(StepMethodState):
    tableau: ButcherTableau
    prev_vf1: Optional[DataTensor]


class ExplicitRungeKutta(StepMethod[ERKState], torch.nn.Module):
    def __init__(self, term: Optional[ODETerm], tableau: ButcherTableau):
        super().__init__()

        self.term = term
        self.tableau = tableau

    def init(
        self,
        y0: DataTensor,
        term: Optional[ODETerm],
        f0: Optional[DataTensor],
        *,
        args: Any,
        t0: float = 0.0,
    ) -> ERKState:
        dtype, device = y0.dtype, y0.device
        if self.tableau.fsal:
            term_ = term
            assert term_ is not None
            t0_v = t0 * torch.ones(y0.shape[0], dtype=dtype, device=device)
            if f0 is None:
                prev_vf1 = term_.vf(t0_v, y0, args)
            else:
                prev_vf1 = f0
        else:
            prev_vf1 = None

        return ERKState(
            tableau=self.tableau.to(
                device=device,
                data_dtype=dtype,
                time_dtype=dtype,
            ),
            prev_vf1=prev_vf1,
        )

    def merge_states(
        self, accept: AcceptTensor, current: ERKState, previous: ERKState
    ) -> ERKState:
        prev_vf1 = previous.prev_vf1
        current_vf1 = current.prev_vf1
        if current_vf1 is None or prev_vf1 is None:
            return current
        else:
            return ERKState(
                current.tableau, torch.where(accept[:, None], current_vf1, prev_vf1)
            )

    def step(
        self,
        term: Optional[ODETerm],
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: ERKState,
        *,
        args: Any,
    ) -> tuple[StepResult, ERKState, Optional[StatusTensor]]:
        term_ = term
        assert term_ is not None
        tableau = state.tableau

        # Convert dt into the data dtype for dtype stability
        dt_data = dt.to(dtype=y0.dtype)

        prev_vf1 = state.prev_vf1
        vf0 = (
            prev_vf1
            if tableau.fsal and prev_vf1 is not None
            else term_.vf(t0, y0, args)
        )
        y_i = y0
        k = vf0.new_empty((tableau.n_stages, vf0.shape[0], vf0.shape[1]))
        k[0] = vf0
        a = tableau.a
        t_nodes = torch.addcmul(t0, tableau.c[:, None], dt)
        for i in range(1, tableau.n_stages):
            y_i = torch.einsum("j, jbf -> bf", a[i, :i], k[:i])
            y_i = torch.addcmul(y0, dt_data[:, None], y_i)
            k[i] = term_.vf(t_nodes[i], y_i, args)

        if tableau.ssal:
            y1 = y_i
        else:
            y1 = y0 + torch.einsum("b, s, sbf -> bf", dt_data, tableau.b, k)
        error_estimate = torch.einsum("b, s, sbf -> bf", dt_data, tableau.b_err, k)

        if tableau.fsal:
            state = ERKState(state.tableau, k[-1])

        return (
            StepResult(y1, error_estimate),
            state,
            None,
        )


class Dopri5(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0],
        a=[
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        b=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
        b_low_order=[
            1951 / 21600,
            0,
            22642 / 50085,
            451 / 720,
            -12231 / 42400,
            649 / 6300,
            1 / 60,
        ],
        b_other=[
            # Coefficients for y at the mid point
            [
                6025192743 / 30085553152 / 2,
                0,
                51252292925 / 65400821598 / 2,
                -2691868925 / 45128329728 / 2,
                187940372067 / 1594534317056 / 2,
                -1776094331 / 19743644256 / 2,
                11237099 / 235043384 / 2,
            ]
        ],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, Dopri5.TABLEAU)

    def convergence_order(self):
        return 5


def rms_norm(y: DataTensor) -> NormTensor:
    """Root mean squared error norm.

    As suggested in [1], Equation (4.11).

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """
    # `vector_norm` autmatically deals with complex vectors correctly
    return torch.linalg.vector_norm(y / math.sqrt(y.shape[1]), ord=2, dim=1)


def max_norm(y: DataTensor) -> NormTensor:
    """Maximums norm."""
    return torch.linalg.vector_norm(y, dim=1, ord=torch.inf)


class AdaptiveStepSizeControllerState(ABC):
    almost_zero: Float
    dt_min: Optional[Float]
    dt_max: Optional[Float]

    def __init__(
        self, almost_zero: Float, dt_min: Optional[Float], dt_max: Optional[Float]
    ):
        assert almost_zero is not None
        self.almost_zero, self.dt_min, self.dt_max = almost_zero, dt_min, dt_max

    @staticmethod
    @abstractmethod
    def default(
        *,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[torch.device | str],
        dt_min: Optional[Tensor],
        dt_max: Optional[Tensor],
    ) -> "AdaptiveStepSizeControllerState":
        pass


class PIDState(AdaptiveStepSizeControllerState):
    def __init__(
        self,
        method_order: int,
        prev_error_ratio: NormTensor,
        prev_prev_error_ratio: NormTensor,
        almost_zero: Float,
        dt_min: Optional[Float] = None,
        dt_max: Optional[Float] = None,
    ):
        assert almost_zero is not None
        super().__init__(almost_zero, dt_min, dt_max)
        self.method_order = method_order
        self.prev_error_ratio = prev_error_ratio
        self.prev_prev_error_ratio = prev_prev_error_ratio

    def update_error_ratios(
        self, prev_error_ratio: NormTensor, prev_prev_error_ratio: NormTensor
    ):
        return PIDState(
            self.method_order,
            prev_error_ratio,
            prev_prev_error_ratio,
            self.almost_zero,
            self.dt_min,
            self.dt_max,
        )

    def __repr__(self):
        return (
            f"PIDState(method_order={self.method_order}, "
            f"prev_error_ratio={self.prev_error_ratio}), "
            f"prev_prev_error_ratio={self.prev_prev_error_ratio}, "
            f"almost_zero={self.almost_zero}, "
            f"dt_min={self.dt_min}), "
            f"dt_max={self.dt_max})"
        )

    @staticmethod
    def default(
        *,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[torch.device | str],
        dt_min: Optional[Tensor],
        dt_max: Optional[Tensor],
    ):
        default_ratio = torch.ones(batch_size, dtype=dtype, device=device)
        # Pre-allocate a fixed, very small number as a lower bound for the error ratio
        if dtype == torch.float16:
            float_min = 1e-5
        elif dtype == torch.float32:
            float_min = 1e-15
        elif dtype == torch.float64:
            float_min = 1e-38
        else:
            raise ValueError("Expected dtype to be float16, float32, or float64")
        almost_zero = torch.tensor(float_min, dtype=dtype, device=device)
        assert almost_zero is not None
        return PIDState(
            method_order, default_ratio, default_ratio, almost_zero, dt_min, dt_max
        )


ControllerStateBound = TypeVar(
    "ControllerStateBound", bound="AdaptiveStepSizeControllerState"
)


class AdaptiveStepSizeController(ABC, Generic[ControllerStateBound], torch.nn.Module):
    rtol: Float
    atol: Float
    norm: NormFunction

    def __init__(self, rtol: float, atol: float, norm: NormFunction):
        super().__init__()
        self.register_buffer("rtol", torch.tensor(rtol))
        self.register_buffer("atol", torch.tensor(atol))
        self.norm = norm

    @abstractmethod
    def dt_factor(self, state: ControllerStateBound, error_ratio: NormTensor) -> Tensor:
        """Compute the growth factor of the timestep."""
        pass

    @abstractmethod
    def initial_state(
        self,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        dt_min: Optional[TimeTensor],
        dt_max: Optional[TimeTensor],
    ) -> ControllerStateBound:
        pass

    @abstractmethod
    def merge_states(
        self,
        running: AcceptTensor,
        current: ControllerStateBound,
        previous: ControllerStateBound,
    ) -> ControllerStateBound:
        pass

    @abstractmethod
    def update_state(
        self,
        state: ControllerStateBound,
        y0: DataTensor,
        dt: Float,
        error_ratio: Optional[NormTensor],
        accept: Optional[Bool],
    ) -> ControllerStateBound:
        pass

    def init(
        self,
        batch_size: int,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device | str] = None,
    ) -> tuple[TimeTensor, ControllerStateBound, Optional[DataTensor]]:
        if dt0 is None:
            raise ValueError("Expected dt0")
        f0 = None
        dtype, device = self.initialize_torch_args(dtype, device)

        dt_min, dt_max = self.get_dt_lims(dtype, device)
        state = self.initial_state(
            method_order, batch_size, dtype, device, dt_min, dt_max
        )
        assert state.almost_zero is not None
        return dt0, state, f0

    def initialize_torch_args(
        self, dtype: Optional[torch.dtype], device: Optional[torch.device | str]
    ) -> tuple[torch.dtype, torch.device]:
        if device is None:
            device = torch.get_default_device()
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        return dtype, device

    def get_dt_lims(self, dtype: torch.dtype, device: torch.device):
        dt_min = self.dt_min
        if dt_min is not None:
            dt_min = torch.tensor(dt_min, dtype=dtype, device=device)
        dt_max = self.dt_max
        if dt_max is not None:
            dt_max = torch.tensor(dt_max, dtype=dtype, device=device)

        return dt_min, dt_max

    def adapt_step_size(
        self,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: ControllerStateBound,
    ) -> tuple[AcceptTensor, TimeTensor, ControllerStateBound, Optional[StatusTensor]]:
        y1, error_estimate = step_result.y, step_result.error_estimate

        if error_estimate is None:
            # If the stepping method could not provide an error estimate, we interpret
            # this as an error estimate that gets the step accepted without changing the
            # step size, i.e. as an error ratio of 1 (disregarding the safety factor).
            return (
                torch.ones_like(dt, dtype=torch.bool),
                dt,
                self.update_state(state, y0, dt, None, None),
                None,
            )

        # Compute error ratio and decide on step acceptance
        error_bounds = torch.add(
            self.atol, torch.maximum(y0.abs(), y1.abs()), alpha=self.rtol
        )
        error = error_estimate.abs()
        # We lower-bound the error ratio by some small number to avoid division by 0 in
        # `dt_factor`.
        error_ratio = torch.maximum(self.norm(error / error_bounds), state.almost_zero)
        accept = error_ratio < 1.0

        # Adapt the step size
        dt_next = dt * self.dt_factor(state, error_ratio).to(dtype=dt.dtype)

        # Check for infinities and NaN
        status = torch.where(
            torch.isfinite(error_ratio),
            status_codes.SUCCESS,
            status_codes.INFINITE_NORM,
        )

        # Enforce the minimum and maximum step size
        dt_min = state.dt_min
        dt_max = state.dt_max
        if dt_min is not None or dt_max is not None:
            abs_dt_next = dt_next.abs()
            dt_next = torch.sign(dt_next) * torch.clamp(abs_dt_next, dt_min, dt_max)
            if dt_min is not None:
                status = torch.where(
                    abs_dt_next < dt_min, status_codes.REACHED_DT_MIN, status
                )

        return (
            accept,
            dt_next,
            self.update_state(state, y0, dt, error_ratio, accept),
            status,
        )


class PIDController(AdaptiveStepSizeController[PIDState]):
    """A PID step size controller.

    The formula for the dt scaling factor with PID control is taken from [1], Equation
    (34).

    References
    ----------
    [1] Söderlind, G. (2003). Digital Filters in Adaptive Time-Stepping. ACM
        Transactions on Mathematical Software, 29, 1–26.
    """

    atol: Float
    rtol: Float
    term: Any
    norm: Callable[[DataTensor], NormTensor]
    dt_min: Float
    dt_max: Float

    def __init__(
        self,
        atol: float,
        rtol: float,
        pcoeff: float,
        icoeff: float,
        dcoeff: float,
        *,
        norm: NormFunction = rms_norm,
        dt_min: Optional[Float] = None,
        dt_max: Optional[Float] = None,
        safety: float = 0.9,
        factor_min: float = 0.2,
        factor_max: float = 10.0,
    ):
        super().__init__(atol, rtol, norm)

        self.dt_min = dt_min
        self.dt_max = dt_max

        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max

    def dt_factor(self, state: PIDState, error_ratio: NormTensor):
        """Compute the growth factor of the timestep."""

        # This is an instantiation of Equation (34) in the Söderlind paper where we have
        # factored out the safety coefficient. I have not found a reference for dividing
        # the PID coefficients by the order of the solver but DifferentialEquations.jl
        # and diffrax both do it, so we do it too. Note that our error ratio is the
        # reciprocal of Söderlind's error ratio (except for the safety factor).
        # Therefore, the factor exponents have the opposite sign from the paper.
        #
        # Interesting thing from the introduction of that paper is that you work with p
        # if you want per-step-error-control and p+1 if you want
        # per-unit-step-error-control where p is the convergence order of the stepping
        # method.
        order = state.method_order
        k_I, k_P, k_D = self.icoeff / order, self.pcoeff / order, self.dcoeff / order

        factor1 = error_ratio ** (-(k_I + k_P + k_D))
        factor2 = state.prev_error_ratio ** (k_P + 2 * k_D)
        factor3 = state.prev_prev_error_ratio**-k_D
        factor = self.safety * factor1 * factor2 * factor3

        return torch.clamp(factor, min=self.factor_min, max=self.factor_max)

    def initial_state(
        self,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        dt_min: Optional[TimeTensor],
        dt_max: Optional[TimeTensor],
    ) -> PIDState:
        return PIDState.default(
            method_order=method_order,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def merge_states(
        self, running: AcceptTensor, current: PIDState, previous: PIDState
    ) -> PIDState:
        return current.update_error_ratios(
            torch.where(running, current.prev_error_ratio, previous.prev_error_ratio),
            torch.where(
                running, current.prev_prev_error_ratio, previous.prev_prev_error_ratio
            ),
        )

    def update_state(
        self,
        state: PIDState,
        y0: DataTensor,
        dt: Float,
        error_ratio: Optional[NormTensor],
        accept: Optional[Bool],
    ) -> PIDState:
        if error_ratio is None:
            return state.update_error_ratios(
                prev_error_ratio=y0.new_ones(dt.shape),
                prev_prev_error_ratio=state.prev_error_ratio,
            )
        else:
            assert accept is not None
            return state.update_error_ratios(
                prev_error_ratio=torch.where(
                    accept, error_ratio, state.prev_error_ratio
                ),
                prev_prev_error_ratio=torch.where(
                    accept, state.prev_error_ratio, state.prev_prev_error_ratio
                ),
            )

    ################################################################################
    # The following methods should be on AdaptiveStepSizeController
    ################################################################################


def step(
    term: ODETerm,
    step_method: StepMethod[StepMethodStateBound],
    step_size_controller: AdaptiveStepSizeController[ControllerStateBound],
    integrator_state: tuple[StepMethodStateBound, ControllerStateBound],
    running: AcceptTensor,
    dt: TimeTensor,
    t: TimeTensor,
    y: DataTensor,
    args: Any,
) -> tuple[
    TimeTensor,
    TimeTensor,
    tuple[StepMethodStateBound, ControllerStateBound],
    DataTensor,
    AcceptTensor,
]:
    """given y(t) and function (t,y,args)->f(t,y,args), return (dt, t, state, y(t+dt), accept)"""
    method_state, controller_state = integrator_state
    step_out = step_method.step(term, y, t, dt, method_state, args=args)
    step_result, method_state_next, method_status = step_out
    controller_out = step_size_controller.adapt_step_size(
        dt, y, step_result, controller_state
    )
    accept, dt_next, controller_state_next, controller_status = controller_out

    # Update the solver state where the step was accepted
    to_update = accept & running
    t = torch.where(to_update, t + dt, t)
    y = torch.where(to_update[:, None], step_result.y, y)
    method_state = step_method.merge_states(to_update, method_state_next, method_state)
    status = method_status
    if status is None:
        status = controller_status
    elif controller_status is not None:
        status = torch.maximum(status, controller_status)

    dt = torch.where(running, dt_next, dt)

    controller_state = step_size_controller.merge_states(
        running, controller_state_next, controller_state
    )

    return dt, t, (method_state, controller_state), y, accept


def default_particle_integrator(
    y0: DataTensor,
    diff: DiffFunction,
    dt0: float,
    *,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    pcoeff: float = 0.2,
    icoeff: float = 0.5,
    dcoeff: float = 0.0,
    args: Optional[tuple] = None,
):
    batch_size, dtype, device = y0.shape[0], y0.dtype, y0.device
    if dtype == torch.float16:
        atol = 1e-4 if atol is None else atol
        rtol = 1e-3 if rtol is None else rtol
    elif dtype == torch.float32:
        atol = 1e-5 if atol is None else atol
        rtol = 1e-3 if rtol is None else rtol
    else:
        atol = 1e-8 if atol is None else atol
        rtol = 1e-5 if rtol is None else rtol
    method = Dopri5()
    term = ODETerm(diff)
    method_state = method.init(y0, term, None, args=args)
    controller = PIDController(atol, rtol, pcoeff, icoeff, dcoeff)
    dt0_v = dt0 * torch.ones(batch_size, dtype=dtype, device=device)
    t0 = torch.zeros(batch_size, dtype=dtype, device=device)
    dt0_v, controller_state, _ = controller.init(
        batch_size, method.convergence_order(), dt0_v, dtype=dtype, device=device
    )

    def step_fcn(
        state: tuple[ERKState, PIDState],
        running: AcceptTensor,
        dt: TimeTensor,
        t: TimeTensor,
        y: DataTensor,
        _args: Any,
    ):
        return step(term, method, controller, state, running, dt, t, y, _args)

    running_0 = torch.ones(batch_size, dtype=torch.bool, device=device)
    state_0 = (method_state, controller_state)
    return torch.compile(step_fcn), state_0, running_0, dt0_v, t0
