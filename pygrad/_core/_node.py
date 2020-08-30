# import abc

# from pygrad._core._config import config


# class _Node(abc.ABC):
#     """Computational graph node.

#     Examples
#     --------
#     >>> import pygrad as gd
#     >>> from pygrad._core._node import _Node
#     >>> a = _Node()
#     >>> b = _Node()
#     >>> with gd.Graph():
#     ...     c = _Node(a, b)
#     >>> a._parents == ()
#     True
#     >>> a._children == [c]
#     True
#     >>> b._parents == ()
#     True
#     >>> b._children == [c]
#     True
#     >>> c._parents == (a, b)
#     True
#     >>> c._children == []
#     True
#     """

#     def __init__(self, *parents, name: str = None):
#         super().__init__()
#         for parent in parents:
#             assert(isinstance(parent, _Node))
#             if config._graph is not None:
#                 parent._children.append(self)
#         self._parents: tuple = parents
#         if name is not None:
#             for ng_char in (',', '(', ')'):
#                 if ng_char in name:
#                     raise ValueError(
#                         f'NG character {ng_char} contained'
#                         f' in arg \'name\', {name}.')
#         self._name: str = name
#         self._children: list = []

#     @property
#     def name(self) -> str:
#         return self._name
