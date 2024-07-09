# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024  Toon Verstraelen
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# --
"""Simple extension of msgpack to support NumPy arrays.

This is a much simpler implementation than msgpack-numpy.
"""

import io
import lzma
from typing import Any, TypeVar

import cattrs
import msgpack
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


# See https://github.com/python-attrs/cattrs/issues/194
CONVERTER = cattrs.Converter()
CONVERTER.register_structure_hook_func(
    lambda t: getattr(t, "__origin__", None) is np.ndarray,
    lambda v, t: np.array([t.__args__[1].__args__[0](e) for e in v]),
)
NUMPY_CODE = 93


def load(path_nmpk_xz: str, cls: type[T]) -> T:
    """Load data from a NumPy Message Pack and structure it with the given cls.

    Parameters
    ----------
    path_nmpk_xz
        Path of an XZ-compressed NumPy Message Pack file.
    cls
        The type to structure into

    Returns
    -------
    result
        An instance cls, loaded from the file.
    """
    with lzma.open(path_nmpk_xz, mode="r") as fh:
        data = msgpack.unpack(fh, ext_hook=_numpy_ext_hook, strict_map_key=False)
    return CONVERTER.structure(data, cls)


def _numpy_ext_hook(code, data) -> NDArray:
    if code == NUMPY_CODE:
        return np.load(io.BytesIO(data), allow_pickle=False)
    return msgpack.ExtType(code, data)


def dump(path_nmpk_xz: str, obj: Any):
    """Load data from a NumPy Message Pack and structure it with the given cls.

    Parameters
    ----------
    path_nmpk_xz
        Path of an XZ-compressed NumPy Message Pack file.
    cls
        The type to structure into

    Returns
    -------
    result
        An instance cls, loaded from the file.
    """
    data = CONVERTER.unstructure(obj)
    with lzma.open(path_nmpk_xz, mode="w") as fh:
        msgpack.pack(data, fh, default=_numpy_default)


def _numpy_default(obj):
    if isinstance(obj, np.ndarray):
        memf = io.BytesIO()
        np.save(memf, obj, allow_pickle=False)
        return msgpack.ExtType(NUMPY_CODE, memf.getvalue())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj
