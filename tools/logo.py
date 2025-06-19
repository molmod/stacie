#!/usr/bin/env python3
"""Python script that draws STACIE's logo.

The design is based on the following ideas:

- S-like shape referring to the name STACIE (and the integral sign, sort of).
- A simple shape, works at different sizes, easily recognizable and memorable.
- Monochrome to make it easily adaptable to different use cases.
  (dark/light background, pdf, print, ...)
- Small SVG file size, with svg.py module.
- No words or letters (other than the S-like shape), so it can be paired with any font.
- Transition from chaos to order, a bit inspired by one of Escher's themes,
  and reflecting the fact that STACIE turns noisy data into useful information.
- Multiple independent time series used as input,
  represented by the aligned dots in the top half of the logo.
- Fun with math: the equation to get the S-shape contains a Lorentzian.

"""

import numpy as np
import svg

# Parameters affecting the logo size and shape
width = 2.5
height = 2.5
yshift = 0.05
pxscale = 60
amax = 1.0 * np.pi
atilt = 0.2 * np.pi
npt = 100
lw = 0.15
spacing = 0.025
w2 = 0.15


def lorentz(a):
    """Compute a flipped Lorentzian and its derivative."""
    p = 4
    r = 1 - 1 / (1 + p * a**2)
    dr = -2 * p * a / (1 + p * a**2) ** 2
    return r, dr


def polar(a, r):
    x = -r * np.cos(a - atilt)
    y = -r * np.sin(a - atilt)
    return np.round(x, 3), np.round(y, 3)


# v is a parameter that controls the radial extent of a series of dots
radii = 0.5 * np.array([lw, lw * 0.7, lw * 0.9, lw])
radii = np.round(radii, 3)

ntrace = len(radii)

vs = np.zeros(ntrace)
vs[1:] = np.cumsum(radii[:-1] + spacing) + np.cumsum(radii[1:])
vs -= vs.mean()
vs += 1

# Top half of the logo: chaos
elements = []
for v, radius in zip(vs, radii, strict=True):
    a = amax
    r = 0
    dr = 0
    while a > 0:
        # Add a point
        r, dr = lorentz(a)
        r *= v
        dr *= v
        x, y = polar(a, r)
        elements.append(svg.Circle(cx=x, cy=y, r=radius))
        # Compute the decrement of the angle to maintain a constant spacing between the points.
        da = (2 * radius + spacing) * r / v / np.sqrt(r**2 + dr**2)
        da = max(da, 0.01 * amax)
        a -= da

# Bottom half of the logo: order
ad = np.linspace(0, amax, npt)
rd, drd = lorentz(ad)
x1, y1 = polar(ad, rd / (1 + w2))
x2, y2 = polar(ad, rd * (1 + w2))
points = -np.block([[x1, x2[::-1]], [y1, y2[::-1]]]).T
poly = svg.Polygon(
    points=points.ravel().tolist(),
    stroke="black",
    stroke_width=lw,
    stroke_linejoin="round",
)
elements.append(poly)

# Black logo (for light background)
vbox = svg.ViewBoxSpec(-width / 2, -height / 2 - yshift, width, height)
canvas = svg.SVG(width=width * pxscale, height=height * pxscale, viewBox=vbox, elements=elements)
with open("../docs/source/static/stacie-logo-black.svg", "w") as f:
    f.write(str(canvas))
    f.write("\n")

# White logo (for dark background)
for element in elements[:-1]:
    element.fill = "white"
elements[-1].fill = "white"
elements[-1].stroke = "white"
with open("../docs/source/static/stacie-logo-white.svg", "w") as f:
    f.write(str(canvas))
    f.write("\n")
