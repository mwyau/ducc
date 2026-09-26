import ctypes
import platform
import sys

import numpy as np


def _native_buffer_format(dtype):
    fmt = memoryview(np.empty(1, dtype=dtype)).format
    if fmt and fmt[0] in "@=<>!":
        prefix, fmt = fmt[0], fmt[1:]
        little = sys.byteorder == "little"
        if (prefix == "<" and not little) or (prefix in ">!" and little):
            return None
    return fmt


HAS_NATIVE_LONGDOUBLE = (
    "ppc64le" not in platform.machine().lower()
    and np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    and np.dtype(np.longdouble).itemsize == ctypes.sizeof(ctypes.c_longdouble)
    and np.dtype(np.clongdouble).itemsize == 2 * ctypes.sizeof(ctypes.c_longdouble)
    and _native_buffer_format(np.longdouble) == "g"
    and _native_buffer_format(np.clongdouble) == "Zg"
)
