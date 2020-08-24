from pygrad.random._categorical import categorical
from pygrad.stats._distribution import Distribution
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy


class Categorical(Distribution):
    r"""Categorical distribution class

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}
    """

    def __init__(self, n_classes: int = None, rv='x', name='Cat'):
        super().__init__(rv=rv, name=name)
        self._n_classes = n_classes

    def forward(self):
        return {'logits': [0] * self._n_classes}

    def _logpdf(self, x, logits):
        return -softmax_cross_entropy(x, logits)

    def _sample(self, logits):
        return categorical(logits)
