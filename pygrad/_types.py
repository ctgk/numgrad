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
        np.int8: Int8,
        np.int16: Int16,
        np.int32: Int32,
        np.int64: Int64,
        np.float16: Float16,
        np.float32: Float32,
        np.float64: Float64,
        np.float128: Float128,
    }
    return dict_.get(type_, None)
