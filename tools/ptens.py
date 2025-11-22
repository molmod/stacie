#!/usr/bin/env python3
"""A Simply Python + SymPy script to derive a rotated pressure tensor."""

import sympy as sp

pxx, pxy, pzx, pyy, pyz, pzz = sp.symbols("pxx pxy pzx pyy pyz pzz")

ptens = sp.Matrix([[pxx, pxy, pzx], [pxy, pyy, pyz], [pzx, pyz, pzz]])
rtens = sp.Matrix(
    [[1, 0, 0], [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)], [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)]]
)

print("Original pressure tensor:")
sp.pprint(ptens, use_unicode=True)

rptens = (rtens * ptens * rtens.T).expand()
print()
print("Rotated pressure tensor:")
sp.pprint(rptens, use_unicode=True)
