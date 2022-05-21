# NumGrad

Simple gradient computation library for Python.

# Getting Started

```bash
pip install numgrad
```

Inspired by [tensorflow](https://www.tensorflow.org/), `numgrad` supports [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) in tensorflow v2 style using original `numpy` and `scipy` functions.

```python
>>> import numgrad as ng
>>> import numpy as np  # Original numpy
>>>
>>> # Pure numpy function
>>> def tanh(x):
...     y = np.exp(-2 * x)
...     return (1 - y) / (1 + y)
...
>>> x = ng.Variable(1)
>>> with ng.Graph() as g:
...     # numgrad patches numpy functions automatically here
...     y = tanh(x)
...
>>> g.backward(y, [x])
(0.419974341614026,)
>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002
0.41997434264973155
```

`numgrad` also supports [jax](https://github.com/google/jax) style automatic differentiation.

```python
>>> import numgrad as ng
>>> import numpy as np  # Original numpy unlike `jax`
>>>
>>> power_derivatives = [lambda a: np.power(a, 5)]
>>> for _ in range(6):
...     power_derivatives.append(ng.grad(power_derivatives[-1]))
...
>>> [f(2) for f in power_derivatives]
[32, 80.0, 160.0, 240.0, 240.0, 120.0, 0.0]
>>> [f(-1) for f in power_derivatives]
[-1, 5.0, -20.0, 60.0, -120.0, 120.0, -0.0]
```

# Contribute

Be sure to run the following command before developing

```bash
$ git clone https://github.com/ctgk/numgrad.git
$ cd numgrad
$ pre-commit install
```
