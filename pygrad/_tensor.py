import typing as tp

import numpy as np

from pygrad._config import config


ArrayLike = tp.Union[np.ndarray, list, tuple]


class Tensor(np.ndarray):
    """Multi-dimensional tensor class.

    Examples
    --------
    >>> a = gd.Tensor([0, 1])
    >>>
    >>> # data type
    >>> a.dtype
    dtype('float64')
    >>> gd.Tensor([1], dtype=np.float32).dtype
    dtype('float32')
    >>>
    >>> # numpy ufunc
    >>> b = a + 1
    >>> type(b)
    <class 'pygrad.Tensor'>
    >>> b
    Tensor([1., 2.])
    """

    def __new__(cls, array: ArrayLike, dtype: type = None) -> 'Tensor':
        """Return tensor.

        Parameters
        ----------
        array : ArrayLike
            Input array
        dtype : type, optional
            One of {`float`, `np.float32`, `np.float64`}, by default None

        Returns
        -------
        Tensor
            Output tensor

        Raises
        ------
        ValueError
            Invalid data type.
        """
        if dtype is None:
            dtype = config.dtype
        if dtype not in (float, np.float32, np.float64):
            raise ValueError(
                'Data type of `Tensor` must be either '
                '`float`, `np.float32`, or `np.float64`, '
                f'not {dtype}')
        return np.asarray(array, dtype=dtype).view(Tensor)

    def __array_ufunc__(  # noqa: D105
        self, ufunc, method, *inputs, out=None, **kwargs,
    ):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Tensor):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Tensor):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asarray(result).view(Tensor) if output is None else output)
            for result, output in zip(results, outputs)
        )
        return results[0] if len(results) == 1 else results


TensorLike = tp.Union[Tensor, ArrayLike]
