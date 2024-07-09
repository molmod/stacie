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
"""Utilities to serialize Stacie objects with Zarr."""

from collections.abc import MutableMapping
from typing import Any, TypeVar

import cattrs
import numpy as np
import zarr

__all__ = ("dump", "load")


# See https://github.com/python-attrs/cattrs/issues/194
CONVERTER = cattrs.Converter()
CONVERTER.register_structure_hook_func(
    lambda t: getattr(t, "__origin__", None) is np.ndarray,
    lambda v, t: np.array([t.__args__[1].__args__[0](e) for e in v]),
)


def dump(obj: Any, store: zarr.storage.BaseStore | MutableMapping | str):
    """Unstructure and serialize into a Zarr store.

    Parameters
    ----------
    object
        The object to store.
    store
        The destination Zarr store.
    """
    data = CONVERTER.unstructure(obj)
    with zarr.open(store, mode="w") as root:
        _dump_nested_dict(root, data)


def _dump_nested_dict(root: zarr.storage.StoreLike, data):
    """Stores a nested dictionary with NumPy arrays in a Zarr group.

    Parameters
    ----------
    root
        The Zarr group to write to.
    data
        The nested dictionary to store.
        Non-array values are stored as metadata.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively store nested dictionaries.
            _dump_nested_dict(root.create_group(key), value)
        elif isinstance(value, np.ndarray):
            # Store NumPy array as Zarr array.
            root[key] = zarr.array(value)
        else:
            root.attrs[key] = value


T = TypeVar("T")


def load(store: zarr.storage.BaseStore | MutableMapping | str, cls: type[T]) -> T:
    """Load data from a Zarr store and structure it with the given cls.

    Parameters
    ----------
    store
        The source Zarr store.
    cls
        The type to structure into

    Returns
    -------
    result
        An instance cls, loaded from the Zarr store.
    """
    with zarr.open(store, mode="r") as root:
        data = _load_nested_dict(root)
    return CONVERTER.structure(data, cls)


def _load_nested_dict(root: zarr.storage.StoreLike):
    """Load a nested set of dictionaries with NumPy arrays from a Zarr store.

    Parameters
    ----------
    root
        The Zarr group to read from.

    Returns
    -------
    data
        A nested dictionary representing the data stored in the Zarr group.
    """
    data = {}
    for key, value in root.arrays():
        data[key] = np.asarray(value)
    for key, value in root.groups():
        data[key] = _load_nested_dict(value)
    for key, value in root.attrs.items():
        data[key] = value  # noqa: PERF403
    return data
