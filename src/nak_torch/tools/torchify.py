from typing import Callable, Any, Optional
import torch
from torch import Tensor


def differentiable_density_factory(
    log_p: Callable[[Any], float],
    grad_log_p: Callable[[Any], Any],
    tensor_transform: Optional[Callable[[Tensor], Any]] = None,
    tensor_inverse_transform: Optional[Callable[[Any], Tensor]] = None,
):

    if tensor_transform is None:
        tensor_transform = lambda x: x  # noqa: E731

    if tensor_inverse_transform is None:
        tensor_inverse_transform = torch.as_tensor

    class Density(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

        @staticmethod
        def forward(ctx, input: torch.Tensor):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache tensors for
            use in the backward pass using the ``ctx.save_for_backward`` method. Other
            objects can be stored directly as attributes on the ctx object, such as
            ``ctx.my_object = my_object``. Check out `Extending torch.autograd <https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd>`_
            for further details.
            """
            ctx.save_for_backward(input)
            return tensor_inverse_transform(log_p(tensor_transform(input)))

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            (input,) = ctx.saved_tensors
            return tensor_inverse_transform(grad_log_p(tensor_transform(input)))

    return Density
