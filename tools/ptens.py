#!/usr/bin/env python3
"""A Simply Python + SymPy script to derive a rotated pressure tensor."""

import sympy as sp

pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz = sp.symbols("pxx pxy pxz pyx pyy pyz pzx pzy pzz")

ptens = sp.Matrix([[pxx, pxy, pxz], [pyx, pyy, pyz], [pzx, pzy, pzz]])
rtens = sp.Matrix(
    [[1, 0, 0], [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)], [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)]]
)

print("Original pressure tensor:")
sp.pprint(ptens, use_unicode=True)

rptens = (rtens * ptens * rtens.T).expand()
print()
print("Rotated pressure tensor:")
sp.pprint(rptens, use_unicode=True)
