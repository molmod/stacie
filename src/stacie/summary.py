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
"""Utility to print a summary of the results object on screen."""

from .estimate import Result
from .plot import UnitConfig

__all__ = ("summarize_results",)


def summarize_results(res: Result | list[Result], uc: UnitConfig | None = None):
    """Return a string summarizing the results object."""
    if isinstance(res, Result):
        res = [res]
    if uc is None:
        uc = UnitConfig()
    texts = []
    for r in res:
        text = GENERAL_TEMPLATE.format(
            r=r,
            uc=uc,
            model=r.props["model"],
            timestep=r.spectrum.timestep / uc.time_unit,
            prefactor=r.spectrum.prefactor / uc.acint_unit,
            acint=r.acint / uc.acint_unit,
            acint_std=r.acint_std / uc.acint_unit,
            corrtime_int=r.corrtime_int / uc.time_unit,
            corrtime_int_std=r.corrtime_int_std / uc.time_unit,
            sensitivity_simulation_time=r.props["sensitivity_simulation_time"] / uc.time_unit,
            sensitivity_block_time=r.props["sensitivity_block_time"] / uc.time_unit,
            npar=len(r.props["pars"]),
            maxdof=r.spectrum.ndofs.max(),
        )
        if "corrtime_exp" in r.props:
            text += EXP_TAIL_TEMPLATE.format(
                uc=uc,
                corrtime_exp=r.props["corrtime_exp"] / uc.time_unit,
                corrtime_exp_std=r.props["corrtime_exp_std"] / uc.time_unit,
                exptail_simulation_time=r.props["exptail_simulation_time"] / uc.time_unit,
                exptail_block_time=r.props["exptail_block_time"] / uc.time_unit,
            )
        texts.append(text)
    return "\n---\n".join(texts)


GENERAL_TEMPLATE = """\
SPECTRUM SETTINGS
    Time step:                     {timestep:{uc.time_fmt}} {uc.time_unit_str}
    Maximum degrees of freedom:    {maxdof}
    Prefactor:                     {prefactor:{uc.acint_fmt}} {uc.acint_unit_str}

MAIN RESULTS
    Autocorrelation integral:      {acint:{uc.acint_fmt}} ± {acint_std:{uc.acint_fmt}} \
{uc.acint_unit_str}
    Integrated correlation time:   {corrtime_int:{uc.time_fmt}} ± {corrtime_int_std:{uc.time_fmt}} \
{uc.time_unit_str}

RECOMMENDED SIMULATION SETTINGS (SENSITIVITY ANALYSIS)
    Simulation time:               {sensitivity_simulation_time:{uc.time_fmt}} {uc.time_unit_str}
    Block time:                    {sensitivity_block_time:{uc.time_fmt}} {uc.time_unit_str}

MODEL {model}
    Number of parameters:          {npar}
    Number points fitted to:       {r.nfit}
"""

EXP_TAIL_TEMPLATE = """\
    Exponential correlation time:  {corrtime_exp:{uc.time_fmt}} ± {corrtime_exp_std:{uc.time_fmt}} \
{uc.time_unit_str}

RECOMMENDED SIMULATION SETTINGS (EXPONENTIAL TAIL MODEL)
    Simulation time:               {exptail_simulation_time:{uc.time_fmt}} {uc.time_unit_str}
    Block time:                    {exptail_block_time:{uc.time_fmt}} {uc.time_unit_str}
"""
