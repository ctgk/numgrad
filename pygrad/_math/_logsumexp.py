import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _logsumexp(
    x: TensorLike,
    *,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
):
    out = sp.logsumexp(x, axis=axis, keepdims=keepdims)
    if isinstance(axis, int):
        axis = (axis,)

    def grad(dout):
        out_ = out  # To prevent UnboundLocalError
        if all((
            isinstance(dout, np.ndarray),
            (not keepdims),
            (axis is not None),
        )):
            axis_positive = []
            for ax in axis:
                if ax < 0:
                    axis_positive.append(x.ndim + ax)
                else:
                    axis_positive.append(ax)
            for ax in sorted(axis_positive):
                dout = np.expand_dims(dout, ax)
                out_ = np.expand_dims(out_, ax)
        dout = np.broadcast_to(dout, x.shape)
        out_ = np.broadcast_to(out_, x.shape)
        return dout * np.exp(x - out_)

    return out, grad


def logsumexp(
    x: TensorLike,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Return natural logarithm of summation of exponentials along give axis.

    .. math::
        f({\boldsymbol x}) = \ln\Sigma_{i=0}^{N-1}e^{x_i}

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axis : tp.Union[int, tp.Tuple[int, ...], None], optional
        Axis to sum along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Tensor
        Natural logarithm of summation of exponentials

    Examples
    --------
    >>> gd.logsumexp([0, 1, -1])
    Tensor(1.40760596)
    """
    return _logsumexp(x, axis=axis, keepdims=keepdims)
