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
>>> g.gradient(y, [x])
(0.419974341614026,)
>>> (tanh(1.0001) - tanh(0.9999)) / 0.0002
0.41997434264973155
```

# Build and Test

# Contribute

Be sure to run the following command before developing

```bash
$ git clone https://github.com/ctgk/numgrad.git
$ cd numgrad
$ pre-commit install
```
