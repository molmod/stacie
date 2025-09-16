<!-- markdownlint-disable line-length -->
# STACIE

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![pytest](https://github.com/molmod/stacie/actions/workflows/pytest.yaml/badge.svg)](https://github.com/molmod/stacie/actions/workflows/pytest.yaml)
[![PyPI](https://img.shields.io/pypi/v/stacie.svg)](https://pypi.python.org/pypi/stacie/)
![Version](https://img.shields.io/pypi/pyversions/stacie.svg)
![License](https://img.shields.io/github/license/molmod/stacie)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15744667.svg)](https://doi.org/10.5281/zenodo.15744667)

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/source/static/github_repo_card_dark.png">
      <source media="(prefers-color-scheme: light)" srcset="docs/source/static/github_repo_card_light.png">
      <img alt="Shows a black logo in light color mode and a white one in dark color mode." src="docs/source/static/github_repo_card_dark.png">
    </picture>
</p>

STACIE is a Python package and algorithm that computes time integrals of autocorrelation functions.
It is primarily designed for post-processing molecular dynamics simulations.
However, it can also be used for more general analysis of time-correlated data.
Typical applications include estimating transport properties and
the uncertainty of averages over time-correlated data, as well as analyzing characteristic timescales.

All information about STACIE can be found in the [documentation](https://molmod.github.io/stacie).

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

> Gözdenur Toraman, Dieter Fauconnier, and Toon Verstraelen
> "STable AutoCorrelation Integral Estimator (STACIE):
> Robust and accurate transport properties from molecular dynamics simulations"
> *Journal of Chemical Information and Modeling* **Article ASAP**, 2025,
> [doi:10.1021/acs.jcim.5c01475](https://doi.org/10.1021/acs.jcim.5c01475),
> [arXiv:2506.20438](https://arxiv.org/abs/2506.20438).
>
> ```bibtex
> @article{Toraman2025,
> title = {STable AutoCorrelation Integral Estimator (STACIE): Robust and accurate transport properties from molecular dynamics simulations},
> url = {https://doi.org/10.1021/acs.jcim.5c01475},
> doi = {10.1021/acs.jcim.5c01475},
> publisher = {Journal of Chemical Information and Modeling},
> author = {G\"{o}zdenur Toraman and Dieter Fauconnier and Toon Verstraelen},
> year = {2025},
> month = {sep}
> }
>
> ```

## Installation

Assuming you have Python and Pip installed,
the following shell command will install STACIE in your Python environment.

```bash
python -m pip install stacie
```
