# STACIE

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![pytest](https://github.com/molmod/stacie/actions/workflows/pytest.yaml/badge.svg)](https://github.com/molmod/stacie/actions/workflows/pytest.yaml)
[![PyPI](https://img.shields.io/pypi/v/stacie.svg)](https://pypi.python.org/pypi/stacie/)
![Version](https://img.shields.io/pypi/pyversions/stacie.svg)
![License](https://img.shields.io/github/license/molmod/stacie)

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/source/static/stacie-logo-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/source/static/stacie-logo-black.svg">
      <img alt="Shows a black logo in light color mode and a white one in dark color mode." src="docs/source/static/stacie-logo-black.svg">
    </picture>
</p>

STACIE stands for the *STable AutoCorrelation Integral Estimator*.
It is a Python package for post-processing molecular dynamics simulations,
but can also be used for more general analysis of time-correlated data.
Typical applications include estimating transport properties,
uncertainties of averages over time-correlated data,
and analysis of characteristic timescales.

All information about STACIE can be found in the [Documentation](https://molmod.github.io/stacie).

## License

STACIE is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

STACIE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.
If not, see [https://www.gnu.org/licenses/](https://www.gnu.org/licenses/).

STACIE's documentation is distributed under the
[Creative Commons CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation

If you use STACIE in your research, please cite the following paper:

> TODO

## Installation

Assuming you have Python and Pip installed,
the following shell command will install STACIE in your Python environment.

```bash
python -m pip install stacie
```
