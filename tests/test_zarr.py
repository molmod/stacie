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
"""Tests for ``stacie.zarr``."""

import attrs
import numpy as np
import zarr
from numpy.testing import assert_equal
from numpy.typing import NDArray
from stacie.zarr import dump, load


@attrs.define
class Foo:
    ar: NDArray[float] = attrs.field()
    name: str = attrs.field()
    other: list[str] = attrs.field()


def test_load_dump_foo():
    foo1 = Foo(np.array([1.0, 2.5, 3.0]), "blabla", ["aaa", "rrr"])
    store = zarr.MemoryStore()
    dump(foo1, store)
    foo2 = load(store, Foo)
    assert isinstance(foo2, Foo)
    assert_equal(foo1.ar, foo2.ar)
    assert foo1.name == foo2.name
    assert foo1.other == foo2.other


@attrs.define
class Bar:
    foo1: Foo = attrs.field()
    foo2: Foo = attrs.field()
    ar: NDArray[int] = attrs.field()


def test_load_dump_bar():
    bar1 = Bar(
        Foo(np.array([1.0, 2.0, 3.0]), "blabla", ["aaa", "rrr"]),
        Foo(np.array([4.0, 11.57]), "wowow", ["bb", "ss"]),
        np.array([3, 7, 9]),
    )
    store = zarr.MemoryStore()
    dump(bar1, store)
    bar2 = load(store, Bar)
    assert isinstance(bar2, Bar)
    assert_equal(bar1.foo1.ar, bar2.foo1.ar)
    assert bar1.foo1.name == bar2.foo1.name
    assert bar1.foo1.other == bar2.foo1.other
    assert_equal(bar1.foo2.ar, bar2.foo2.ar)
    assert bar1.foo2.name == bar2.foo2.name
    assert bar1.foo2.other == bar2.foo2.other
    assert_equal(bar1.ar, bar2.ar)
