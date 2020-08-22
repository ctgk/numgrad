import numpy as np


class DataType:
    pass


class Int8(np.int8, DataType):
    pass


class Int16(np.int16, DataType):
    pass


class Int32(np.int32, DataType):
    pass


class Int64(np.int64, DataType):
    pass


class Float16(np.float16, DataType):
    pass


class Float32(np.float32, DataType):
    pass


class Float64(np.float64, DataType):
    pass


class Float128(np.float128, DataType):
    pass


def _to_pygrad_type(type_):
    dict_ = {
        np.dtype('int8'): Int8,
        np.dtype('int16'): Int16,
        np.dtype('int32'): Int32,
        np.dtype('int64'): Int64,
        np.dtype('float16'): Float16,
        np.dtype('float32'): Float32,
        np.dtype('float64'): Float64,
        np.dtype('float128'): Float128,
    }
    return dict_.get(type_, None)


def _is_int(type_):
    return type_ in (Int8, Int16, Int32, Int64)


def _is_float(type_):
    return type_ in (Float16, Float32, Float64, Float128)
