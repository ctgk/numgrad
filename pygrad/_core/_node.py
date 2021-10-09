import abc
import typing as tp


class _Node(abc.ABC):

    def __init__(
        self,
        parents: tuple = tuple(),
        name: str = None,
    ) -> None:
        self._check_parents_type(parents)
        self._check_name_type_and_validtiy(name)
        self._parents: tp.Tuple[_Node] = parents
        self._name: tp.Union[None, str] = name

    @property
    def parents(self):
        return self._parents

    @property
    def name(self):
        return self._name

    @staticmethod
    def _check_parents_type(parents):
        if not isinstance(parents, tuple):
            raise TypeError(
                '`parents` must be an instance of tuple, '
                f'not {type(parents)}.',
            )
        for o in parents:
            if not isinstance(o, _Node):
                raise TypeError(
                    'Parent of a node must be an instance of node, '
                    f'not {type(o)}.',
                )

    @staticmethod
    def _check_name_type_and_validtiy(name):
        if name is None:
            return
        if not isinstance(name, str):
            raise TypeError(
                '`name` must be an instance of str, '
                f'not {type(name)}.',
            )
        for ng_char in (',', '(', ')'):
            if ng_char in name:
                raise ValueError(f'`name` contains NG character {ng_char}.')
