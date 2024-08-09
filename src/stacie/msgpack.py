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
"""Simple extension of ``msgpack`` to support NumPy arrays.


This is a much simpler `NumPy <https://numpy.org>`_ extensions
of `msgpack <https://pypi.org/project/msgpack/>`_
than `msgpack-numpy <https://pypi.org/project/msgpack-numpy/>`_.

We recommend storing spectra and other results with the serialization functions below
instead of using Python pickle files.
Pickle files are unfit for long-term data preservation because they can only be read again
if your future Python version is compatible with the Python packages used to create the files.
`MessagePack <https://msgpack.org/>`_ files can be used in almost all programming languages.
This NumPy extension is fairly straightforward:
arrays are stored as binary blobs (extension type 93) in NPY format, with ``allow_pickle=False``.

Arbitrary objects can be stored. They are unstructered with `cattrs <https://catt.rs/>`_.
Files are compressed with ``xz`` by default (not optional).

Keep in mind that individual arrays cannot exceed the 4GB size limit imposed by the msgpack format.
"""

import io
import lzma
from typing import Any, TypeVar

import cattrs
import msgpack
import numpy as np
from numpy.typing import NDArray

__all__ = ("load", "dump")


T = TypeVar("T")


# See https://github.com/python-attrs/cattrs/issues/194
CONVERTER = cattrs.Converter()
CONVERTER.register_structure_hook_func(
    lambda t: getattr(t, "__origin__", None) is np.ndarray,
    lambda v, t: np.array([t.__args__[1].__args__[0](e) for e in v]),
)
NUMPY_CODE = 93


def load(path_nmpk_xz: str, cls: type[T]) -> T:
    """Load data from a NumPy Message Pack and structure it with the given ``cls``.

    Parameters
    ----------
    path_nmpk_xz
        Path of an XZ-compressed NumPy Message Pack file.
    cls
        The type to structure into.

    Returns
    -------
    result
        An instance of ``cls``, loaded from the file.
    """
    with lzma.open(path_nmpk_xz, mode="r") as fh:
        data = msgpack.unpack(fh, ext_hook=_numpy_ext_hook, strict_map_key=False)
    return CONVERTER.structure(data, cls)


def _numpy_ext_hook(code, data) -> NDArray:
    if code == NUMPY_CODE:
        return np.load(io.BytesIO(data), allow_pickle=False)
    return msgpack.ExtType(code, data)


def dump(path_nmpk_xz: str, obj: Any):
    """Unstructure an object and dump the data to a NumPy Message Pack file.

    Parameters
    ----------
    path_nmpk_xz
        Path of an XZ-compressed NumPy Message Pack file.
    obj
        The type to structure into.
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
