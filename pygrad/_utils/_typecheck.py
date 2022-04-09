import functools
import inspect
from itertools import chain
import typing as tp


def _is_union_type(tp):
    return repr(tp).startswith('typing.Union')


def _is_list_type(tp):
    return "typing.List[" == repr(tp)[:12] and "]" == repr(tp)[-1]


def _is_tuple_type(tp):
    return "typing.Tuple[" == repr(tp)[:13] and "]" == repr(tp)[-1]


def _is_dict_type(tp):
    return "typing.Dict[" == repr(tp)[:12] and ']' == repr(tp)[-1]


def _is_subclass_type(tp):
    return "typing.Type[" == repr(tp)[:12] and "]" == repr(tp)[-1]


def _is_optional_type(tp):
    return "typing.Optional[" == repr(tp)[:16]


def _typecheck_arg(
        obj: object,
        type_: tp.Union[type, tp.Iterable[type]],
        exclude_types: tp.Tuple[tp.Type]) -> bool:
    if type_ in exclude_types:
        return True
    elif isinstance(type_, (tuple, list)):
        return any(_typecheck_arg(obj, t, exclude_types) for t in type_)
    elif _is_optional_type(type_):
        if obj is None:
            return True
        else:
            return _typecheck_arg(obj, type_.__args__, exclude_types)
    elif _is_union_type(type_):
        return _typecheck_arg(obj, type_.__args__, exclude_types)
    elif _is_list_type(type_):
        if isinstance(obj, list):
            return all(
                _typecheck_arg(o, type_.__args__, exclude_types) for o in obj)
        else:
            return False
    elif _is_tuple_type(type_):
        if isinstance(obj, tuple):
            return all(
                _typecheck_arg(o, type_.__args__, exclude_types) for o in obj)
        else:
            return False
    elif _is_dict_type(type_):
        return isinstance(obj, dict) and all(
            _typecheck_arg(k, type_.__args__[0], exclude_types)
            and _typecheck_arg(v, type_.__args__[1], exclude_types)
            for k, v in obj.items())
    elif _is_subclass_type(type_):
        if inspect.isclass(obj):
            return issubclass(obj, type_.__args__)
        else:
            return False
    elif type_ is callable:
        return callable(obj)
    else:
        try:
            return isinstance(obj, type_)
        except Exception as e:
            print(obj, type_)
            raise e


def _typecheck(
    exclude_args: tp.Tuple[str] = (),
    exclude_types: tp.Tuple[tp.Type] = (),
) -> callable:

    def _wrapper(func: callable) -> callable:
        argspec = inspect.getfullargspec(func)
        arg_names = argspec.args + argspec.kwonlyargs
        annotations = argspec.annotations
        for exclude_ in exclude_args:
            if exclude_ not in arg_names:
                raise ValueError(
                    f'No argument named "{exclude_}" in {func.__name__}().')

        @functools.wraps(func)
        def func_with_typechecked_args(*args, **kwargs):
            for name, arg in chain(zip(arg_names, args), kwargs.items()):
                if ((name in annotations)
                        and (name not in exclude_args)
                        and (not _typecheck_arg(
                            arg, annotations[name], exclude_types))):
                    raise TypeError(
                        f"{func.__name__}() argument '{name}' must be "
                        f"{annotations[name]}, not {type(arg)}.")
            return func(*args, **kwargs)

        func_with_typechecked_args.__signature__ = inspect.signature(func)
        return func_with_typechecked_args

    return _wrapper


if __name__ == "__main__":
    _typecheck_arg({'x': 1}, tp.Dict[str, int], ())
