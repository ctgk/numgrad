import abc
import typing as tp


class _Node(abc.ABC):

    def __init__(
        self,
        name: str = None,
    ) -> None:
        self._name: tp.Union[None, str] = self._check_name_arg(name)

    @property
    def name(self) -> tp.Union[None, str]:
        return self._name

    @staticmethod
    def _check_name_arg(name):
        if name is None:
            return None
        if not isinstance(name, str):
            raise TypeError(
                '`name` must be an instance of str, '
                f'not {type(name)}.',
            )
        for ng_char in (',', '(', ')'):
            if ng_char in name:
                raise ValueError(f'`name` contains NG character {ng_char}.')
        return name
