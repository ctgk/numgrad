from collections.abc import Iterable
import functools
import inspect
from itertools import chain
import typing as tp


def _is_union_type(tp):
    return "typing.Union[" == repr(tp)[:13] and "]" == repr(tp)[-1]


def _is_iterable_type(tp):
    return "typing.Iterable[" == repr(tp)[:16] and "]" == repr(tp)[-1]


def _is_subclass_type(tp):
    return "typing.Type[" == repr(tp)[:12] and "]" == repr(tp)[-1]


def _typecheck_arg(obj: object, tp: type) -> bool:
    if isinstance(tp, Iterable):
        return any(_typecheck_arg(obj, t) for t in tp)
    elif _is_union_type(tp):
        return _typecheck_arg(obj, tp.__args__)
    elif _is_iterable_type(tp):
        if isinstance(obj, Iterable):
            return all(_typecheck_arg(o, tp.__args__) for o in obj)
        else:
            return False
    elif _is_subclass_type(tp):
        if inspect.isclass(obj):
            return issubclass(obj, tp.__args__)
        else:
            return False
    else:
        return isinstance(obj, tp)


def _typecheck(exclude: tp.Tuple[str] = ()) -> callable:

    def _wrapper(func: callable) -> callable:
        argspec = inspect.getfullargspec(func)
        arg_names = argspec.args
        annotations = argspec.annotations
        for exclude_ in exclude:
            if exclude_ not in arg_names:
                raise ValueError(
                    f'No argument named "{exclude_}" in {func.__name__}().')

        @functools.wraps(func)
        def func_with_typechecked_args(*args, **kwargs):
            for name, arg in chain(zip(arg_names, args), kwargs.items()):
                if ((name in annotations)
                        and (name not in exclude)
                        and (not _typecheck_arg(arg, annotations[name]))):
                    raise TypeError(
                        f"{func.__name__}() argument '{name}' must be "
                        f"{annotations[name]}, not {type(arg)}.")
            return func(*args, **kwargs)

        func_with_typechecked_args.__signature__ = inspect.signature(func)
        return func_with_typechecked_args

    return _wrapper
