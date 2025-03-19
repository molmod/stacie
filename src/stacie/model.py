# Stacie is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the Stacie Python Package.
# See the CONTRIBUTORS.md file in the project root for a full list of contributors.
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
"""Models to fit the low-frequency part of the spectrum."""

import attrs
import numpy as np
from numpy.typing import NDArray

__all__ = ("ChebyshevModel", "ExpTailModel", "PadeModel", "SpectrumModel", "guess")


@attrs.define
class SpectrumModel:
    """Abstract base class for spectrum models.

    Subclasses must override the attribute ``name``
    and the methods ``bounds``, ``guess``, ``compute`` and ``derive_props``.
    """

    scales: dict[str, float] = attrs.field(factory=dict, init=False)
    """A dictionary with essential scale information for the parameters and the cost function."""

    @property
    def name(self):
        raise NotImplementedError

    def bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        raise NotImplementedError

    @property
    def npar(self):
        """Return the number of parameters."""
        raise NotImplementedError

    def valid(self, pars) -> bool:
        """Return ``True`` when the parameters are within the feasible region."""
        return all(pmin < par < pmax for (pmin, pmax), par in zip(self.bounds(), pars, strict=True))

    def which_invalid(self, pars) -> NDArray[bool]:
        """Return a boolean mask for the parameters outside the feasible region."""
        return np.array(
            [
                pmin >= par or par >= pmax
                for (pmin, pmax), par in zip(self.bounds(), pars, strict=True)
            ]
        )

    def configure_scales(
        self, timestep: float, freqs: NDArray[float], amplitudes: NDArray[float]
    ) -> NDArray[float]:
        """Store essential scale information in the ``scales`` attribute.

        Other methods may access this information,
        so this method should be called before performing any computations.
        """
        self.scales = {
            "freq_small": freqs[1],
            "freq_scale": freqs[-1],
            "time_scale": 1 / freqs[-1],
            "amp_scale": np.median(abs(amplitudes[amplitudes != 0])),
        }

    @property
    def par_scales() -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        raise NotImplementedError

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters.

        The returned parameters cannot be solved with the solve_linear method.
        Models are free to decide which parameters can be solved with linear regression.
        For example, some non-linear parameters may be solved with a linear regression
        after rewriting the regression problem in a different form.
        """
        raise NotImplementedError

    def sample_nonlinear_pars(
        self,
        rng: np.random.Generator,
        budget: int,
    ) -> NDArray[float]:
        """Return samples of the nonlinear parameters.

        Parameters
        ----------
        rng
            The random number generator.
        budget
            The number of samples to generate.
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        par_scales
            The scales of the parameters and the cost function.

        Returns
        -------
        samples
            The samples of the nonlinear parameters, array with shape ``(budget, num_nonlinear)``,
            where ``num_nonlinear`` is the number of nonlinear parameters.
        """
        raise NotImplementedError

    def compute(
        self, timestep: float, freqs: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """Compute the amplitudes of the spectrum model.

        Parameters
        ----------
        timestep
            The time step of the sequences used to compute the spectrum.
            It may be used to convert the frequency to the dimensionless
            normalized frequency :math:`2\\pi h f=2\\pi k/N`,
            where :math:`h` is the timestep,
            :math:`f` is the frequency,
            :math:`k` is the frequency index in the discrete Fourier transform,
            and :math:`N` is the number of samples in the input time series.
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        pars
            The parameters.
        deriv
            The maximum order of derivatives to compute: 0, 1 or 2.

        Returns
        -------
        results
            A results list, index corresponds to order of derivative.
            The shape of the arrays in the results list is as follows:

            - For ``deriv=0``, the shape is ``(len(freqs),)``.
            - For ``deriv=1``, the shape is ``(len(pars), len(freqs))``.
            - For ``deriv=2``, the shape is ``(len(pars), len(pars), len(freqs))``
        """
        raise NotImplementedError

    def solve_linear(
        self,
        timestep: float,
        freqs: NDArray[float],
        ndofs: NDArray[float],
        amplitudes: NDArray[float],
        nonlinear_pars: NDArray[float],
    ) -> NDArray[float]:
        """Use linear linear regression to solve a subset of the parameters.

        The default implementation in the base class assumes that the linear parameters
        are genuinly linear without rewriting the regression problem in a different form.

        Parameters
        ----------
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        amplitudes
            The amplitudes of the spectrum.
        ndofs
            The number of degrees of freedom at each frequency.
        timestep
            The time step of the sequences used to compute the spectrum.
        nonlinear_pars
            The values of the nonlinear parameters for which the basis functions are computed.

        Returns
        -------
        linear_pars
            The solved linear parameters.
        amplitudes_model
            The model amplitudes computed with the solved parameters.
        """
        nonlinear_mask = self.get_par_nonlinear()
        pars = np.ones(self.npar)
        pars[nonlinear_mask] = nonlinear_pars
        basis = self.compute(timestep, freqs, pars, 1)[1][~nonlinear_mask]
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)
        linear_pars = np.linalg.lstsq(
            (basis / amplitudes_std).T,
            amplitudes / amplitudes_std,
            # For compatibility with numpy < 2.0
            rcond=-1,
        )[0]
        amplitudes_model = np.dot(linear_pars, basis)
        return linear_pars, amplitudes_model

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float], pars_sensitivity: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters.

        Parameters
        ----------
        pars
            The parameters.
        covar
            The covariance matrix of the parameters.
        pars_sensitivity
            The sensitivity of the parameters to the empirical spectrum.
            This is an array with shape ``(len(pars), len(freqs))``.

        Returns
        -------
        props
            A dictionary with additional properties,
            whose calculation requires model-specific knowledge.
            This includes:

            - The estimate of the autocorrelation integral, its variance and standard deviation.
            - The estimate of the exponential correlation time, its variance and standard deviation.
              (Set to ``np.nan`` if not applicable.)
            - The sensitivity of the autocorrelation integral to the empirical spectrum.
        """
        raise NotImplementedError


@attrs.define
class ExpTailModel(SpectrumModel):
    r"""The Exponential Tail model for the spectrum.

    This model is derived under the following assumptions:

    - RFFT is used to compute the spectrum.
    - The autocorrelation consists of two contributions:
        - A quickly decaying part with no specific functional form,
          but confined within a small time lag domain centered at t=0.
        - An slow exponentially decaying part with a yet unknown characteristic time scale.

    The mathematic form is:

    .. math::

        C^\text{exp-tail}_k
            \approx
            a_\text{short} + a_\text{tail} \frac{1-r}{1+r} \left(
                2\frac{1 - r\cos(2\pi k/N)}{1 - 2r\cos(2\pi k/N) + r^2} - 1
            \right)

    with

    .. math::

        r = \exp\left(-\frac{h}{\tau_\text{exp}}\right)

    where :math:`h` is the time step and :math:`k` is the index of the RFFT vector.
    (Note that in the implementation :math:`2\pi k/N` is computed as :math:`2\pi h f`,
    where :math:`f` is the frequency.)

    The three parameters of the model in this implementation are:

    - :math:`a_\text{short}`:
      The prefactor for the short-term (white noise) component.
    - :math:`a_\text{tail}`:
      The prefactor for the exponential tail component.
    - :math:`\tau_\text{exp}`:
      The correlation time of the exponential tail
      {cite:p}`sokal_1997_monte`.
    """

    @property
    def name(self):
        return "exptail"

    def bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(0, np.inf), (0, np.inf), (0, np.inf)]

    @property
    def npar(self):
        """Return the number of parameters."""
        return 3

    @property
    def par_scales(self) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        return np.array(
            [self.scales["amp_scale"], self.scales["amp_scale"], self.scales["time_scale"]]
        )

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters."""
        return np.array([False, False, True])

    def sample_nonlinear_pars(
        self,
        rng: np.random.Generator,
        budget: int,
    ) -> NDArray[float]:
        """Return samples of the nonlinear parameters."""
        corrtime_min = 0.1 / self.scales["freq_scale"]
        corrtime_max = 1 / self.scales["freq_small"]
        return np.exp(rng.uniform(np.log(corrtime_min), np.log(corrtime_max), (budget, 1)))

    def compute(
        self, timestep: float, freqs: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")
        acint_short, acint_tail, corrtime = pars
        r = np.exp(-timestep / corrtime)
        cs = np.cos(2 * np.pi * timestep * freqs)
        denom = r**2 - 2 * r * cs + 1
        tail_model = (1 - r) ** 2 / denom
        results = [acint_short + acint_tail * tail_model]
        if deriv >= 1:
            tail_model_diff_r = -2 * (cs - 1) * (r**2 - 1) / denom**2
            r_diff_ct = r * timestep / corrtime**2
            tail_model_diff_ct = tail_model_diff_r * r_diff_ct
            results.append(
                np.array([np.ones(len(freqs)), tail_model, acint_tail * tail_model_diff_ct])
            )
        if deriv >= 2:
            tail_model_diff_r_r = 4 * (cs - 1) * (r**3 - 3 * r + 2 * cs) / denom**3
            r_diff_ct_ct = (1 - 2 * corrtime / timestep) * r * (timestep / corrtime**2) ** 2
            tail_model_diff_ct_ct = (
                tail_model_diff_r_r * r_diff_ct**2 + tail_model_diff_r * r_diff_ct_ct
            )
            results.append(
                np.array(
                    [
                        [np.zeros(len(freqs)), np.zeros(len(freqs)), np.zeros(len(freqs))],
                        [
                            np.zeros(len(freqs)),
                            np.zeros(len(freqs)),
                            tail_model_diff_ct,
                        ],
                        [
                            np.zeros(len(freqs)),
                            tail_model_diff_ct,
                            acint_tail * tail_model_diff_ct_ct,
                        ],
                    ]
                )
            )
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float], pars_sensitivity: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        acint = pars[:2].sum()
        acint_var = covar[:2, :2].sum()
        corrtime = pars[2]
        corrtime_var = covar[2, 2]
        return {
            "model": self.name,
            "acint": acint,
            "acint_var": acint_var,
            "acint_std": np.sqrt(acint_var) if acint_var >= 0 else np.inf,
            "acint_sensitivity": pars_sensitivity[:2].sum(axis=0),
            "corrtime_exp": corrtime,
            "corrtime_exp_var": corrtime_var,
            "corrtime_exp_std": (np.sqrt(corrtime_var) if corrtime_var >= 0 else np.inf),
            "exptail_block_time": corrtime * np.pi / 20,
            "exptail_simulation_time": 20 * corrtime * np.pi,
        }


@attrs.define
class ChebyshevModel(SpectrumModel):
    """A linear combination of Chebyshev polynomials."""

    degree: int = attrs.field(converter=int, validator=attrs.validators.ge(0))
    """The highest degree of the polynomials included in the model.

    If even is ``True``, only even polynomials are included.
    For example, if degree is 3 and even is ``True``,
    the model includes the polynomials :math:`T_0`, :math:`T_2`.
    """

    even: bool = attrs.field(converter=bool, default=False)
    """If ``True``, only even polynomials are included. If ``False``, all terms are included."""

    @property
    def name(self):
        return f"evencheb({self.degree})" if self.even else f"cheb({self.degree})"

    def bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(-np.inf, np.inf)] * self.npar

    @property
    def npar(self):
        """Return the number of parameters."""
        return (self.degree + 2) // 2 if self.even else self.degree + 1

    @property
    def par_scales(self) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        return np.full(self.npar, self.scales["amp_scale"])

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters."""
        return np.zeros(self.npar, dtype=bool)

    def compute(
        self, timestep: float, freqs: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")

        # Construct a basis of Chebyshev polynomials.
        freq_scale = self.scales["freq_scale"]
        basis = [np.ones(len(freqs))]
        if self.degree > 0:
            if self.even:
                basis.append(freqs / freq_scale)
            else:
                # Reverse the frequency axis, so all basis functions are 1 at freq 0.
                basis.append(1 - 2 * freqs / freq_scale)
        for _ in range(2, self.degree + 1):
            basis.append(2 * basis[1] * basis[-1] - basis[-2])
        basis = np.array(basis)
        if self.even:
            basis = basis[::2]
            # Flip the signs of every other basis function, say they are all 1 at freq 0.
            basis *= 1 - 2 * (np.arange(len(basis)).reshape(-1, 1) % 2)

        # Compute model amplitudes and derivatives.
        results = [np.dot(pars, basis)]
        if deriv >= 1:
            results.append(basis)
        if deriv >= 2:
            results.append(np.zeros((self.npar, self.npar, len(freqs))))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float], pars_sensitivity: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        acint = pars.sum()
        acint_var = covar.sum()
        return {
            "model": self.name,
            "acint": acint,
            "acint_var": acint_var,
            "acint_std": np.sqrt(acint_var) if acint_var >= 0 else np.inf,
            "acint_sensitivity": pars_sensitivity.sum(axis=0),
        }


@attrs.define
class PadeModel(SpectrumModel):
    """A rational function model for the spectrum, a.k.a. a Padé approximation."""

    numer_degrees: list[int] = attrs.field(converter=list)
    """The degrees of the mononomials in the numerator."""

    @numer_degrees.validator
    def _validate_num_degrees(self, attribute, value):
        if not all(isinstance(degree, int) and degree >= 0 for degree in value):
            raise ValueError("All numer_degrees must be non-negative, got {value}.")
        if len(value) == 0:
            raise ValueError("The list of numer_degrees must not be empty.")
        if len(value) != len(set(value)):
            raise ValueError("The list of numer_degrees must not contain duplicates.")

    denom_degrees: list[int] = attrs.field(converter=list)
    """The degrees of the mononomials in the denominator.

    Note that the leading term is always 1, and there is no need to include
    degree zero.
    """

    @denom_degrees.validator
    def _validate_num_degrees(self, attribute, value):
        if not all(isinstance(degree, int) and degree >= 1 for degree in value):
            raise ValueError("All denom_degrees must be structky positive, got {value}.")
        if len(value) == 0:
            raise ValueError("The list of denom_degrees must not be empty.")
        if len(value) != len(set(value)):
            raise ValueError("The list of denom_degrees must not contain duplicates.")

    @property
    def name(self):
        return f"pade({self.numer_degrees},{self.denom_degrees})"

    def bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(-np.inf, np.inf)] * len(self.numer_degrees) + [(0, np.inf)] * len(
            self.denom_degrees
        )

    @property
    def npar(self):
        """Return the number of parameters."""
        return len(self.numer_degrees) + len(self.denom_degrees)

    @property
    def par_scales(self) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        return np.concatenate(
            [
                np.full(len(self.numer_degrees), self.scales["amp_scale"]),
                np.ones(len(self.denom_degrees)),
            ]
        )

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters."""
        return np.zeros(self.npar, dtype=bool)

    def compute(
        self, timestep: float, freqs: NDArray[float], pars: NDArray[float], deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")
        npar_n = len(self.numer_degrees)
        npar_d = len(self.denom_degrees)
        if len(pars) != npar_n + npar_d:
            raise ValueError("The number of parameters does not match the model.")

        # Construct two bases of monomials.
        x = freqs / self.scales["freq_scale"]
        basis_n = np.power.outer(x, self.numer_degrees).T
        basis_d = np.power.outer(x, self.denom_degrees).T
        pars_n = pars[:npar_n]
        pars_d = pars[npar_n:]

        # Compute model amplitudes and derivatives.
        num = np.dot(pars_n, basis_n)
        denom = 1 + np.dot(pars_d, basis_d)
        results = [num / denom]
        if deriv >= 1:
            block_n = basis_n / denom
            block_d = -results[0] * basis_d / denom
            results.append(np.concatenate([block_n, block_d]))
        if deriv >= 2:
            block_nn = np.zeros((npar_n, npar_n, len(freqs)))
            block_nd = np.einsum("if,jf->ijf", block_n, -basis_d / denom)
            block_dn = block_nd.transpose(1, 0, 2)
            block_dd = np.einsum("f,if,jf->ijf", 2 * results[0], basis_d / denom, basis_d / denom)
            results.append(
                np.concatenate(
                    [
                        np.concatenate([block_nn, block_nd], axis=1),
                        np.concatenate([block_dn, block_dd], axis=1),
                    ]
                )
            )
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def solve_linear(
        self,
        timestep: float,
        freqs: NDArray[float],
        ndofs: NDArray[float],
        amplitudes: NDArray[float],
        nonlinear_pars: NDArray[float],
    ) -> NDArray[float]:
        """Use linear linear regression to solve a subset of the parameters.

        This is a specialized implementation that rewrites the regersion problem
        in a different form to solve all parameters with a linear regression.
        """
        if len(nonlinear_pars) != 0:
            raise ValueError("The number of nonlinear parameters must be exactly 0.")
        x = freqs / freqs[-1]
        basis_n = np.power.outer(x, self.numer_degrees).T
        basis_d = np.power.outer(x, self.denom_degrees).T
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)
        part_n = basis_n / amplitudes_std
        part_d = -basis_d * (amplitudes / amplitudes_std)
        design_matrix = np.concatenate([part_n, part_d]).T
        expected_values = amplitudes / amplitudes_std
        pars = np.linalg.lstsq(design_matrix, expected_values, rcond=-1)[0]
        npar_n = len(self.numer_degrees)
        pars_n = pars[:npar_n]
        pars_d = pars[npar_n:]
        amplitudes_model = np.dot(pars_n, basis_n) / (1 + np.dot(pars_d, basis_d))
        return pars, amplitudes_model

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float], pars_sensitivity: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return additional properties derived from model-specific parameters."""
        acint = pars[0]
        acint_var = covar[0, 0]
        return {
            "model": self.name,
            "acint": acint,
            "acint_var": acint_var,
            "acint_std": np.sqrt(acint_var) if acint_var >= 0 else np.inf,
            "acint_sensitivity": pars_sensitivity[0],
        }


def guess(
    model: SpectrumModel,
    timestep: float,
    freqs: NDArray[float],
    ndofs: NDArray[float],
    amplitudes: NDArray[float],
    par_scales: NDArray[float],
    rng: np.random.Generator,
    nonlinear_budget: int,
):
    """Guess initial values of the parameters for a model.

    Parameters
    ----------
    model
        The model for which the parameters are guessed.
    freqs
        The frequencies for which the model spectrum amplitudes are computed.
    amplitudes
        The amplitudes of the spectrum.
    ndofs
        The number of degrees of freedom at each frequency.
    timestep
        The time step of the sequences used to compute the spectrum.
    par_scales
        The scales of the parameters and the cost function, obtained from the model.
    rng
        The random number generator.
    nonlinear_budget
        The number of samples of the nonlinear parameters is computed as
        ``nonlinear_budget ** num_nonlinear``, where ``num_nonlinear`` is the number
        of nonlinear parameters.

    Returns
    -------
    pars
        An initial guess of the parameters.
    """
    if not isinstance(nonlinear_budget, int) or nonlinear_budget < 1:
        raise ValueError("Argument nonlinear_budget must be a strictly positive integer.")

    # Get the mask for the nonlinear parameters
    nonlinear_mask = model.get_par_nonlinear()
    num_nonlinear_pars = nonlinear_mask.sum()

    # If there are no nonlinear parameters, we can directly guess the linear parameters.
    if num_nonlinear_pars == 0:
        return _guess_linear(
            model, [], nonlinear_mask, timestep, freqs, amplitudes, ndofs, par_scales
        )[1]

    # Otherwise, we need to sample the nonlinear parameters and guess the linear parameters.
    nonlinear_samples = model.sample_nonlinear_pars(rng, nonlinear_budget**num_nonlinear_pars)
    best = None
    for nonlinear_pars in nonlinear_samples:
        cost, pars = _guess_linear(
            model, nonlinear_pars, nonlinear_mask, timestep, freqs, amplitudes, ndofs, par_scales
        )
        if best is None or best[0] > cost:
            best = cost, pars
    return best[1]


def _guess_linear(
    model: SpectrumModel,
    nonlinear_pars: NDArray[float],
    nonlinear_mask: NDArray[bool],
    timestep: float,
    freqs: NDArray[float],
    amplitudes: NDArray[float],
    ndofs: NDArray[float],
    par_scales: NDArray[float],
) -> tuple[float, NDArray[float]]:
    """Guess initial values of the linear parameters for a model.

    Parameters
    ----------
    model
        The model for which the parameters are guessed.
    nonlinear_pars
        The values of the nonlinear parameters.
    nonlinear_mask
        A boolean mask for the nonlinear parameters.
    timestep
        The time step of the sequences used to compute the spectrum.
    freqs
        The frequencies for which the model spectrum amplitudes are computed.
    amplitudes
        The amplitudes of the spectrum.
    ndofs
        The number of degrees of freedom at each frequency.
    par_scales
        The scales of the parameters and the cost function, obtained from the model.

    Returns
    -------
    cost
        The cost of the guess.
    pars
        An initial guess of the parameters.
    """
    # Perform a weighted least squares fit to guess the linear parameters.
    linear_pars, amplitudes_model = model.solve_linear(
        timestep, freqs, ndofs, amplitudes, nonlinear_pars
    )

    # Combine the linear and nonlinear parameters
    pars = np.zeros(model.npar)
    pars[nonlinear_mask] = nonlinear_pars
    pars[~nonlinear_mask] = linear_pars

    # Fix invalid guesses
    invalid_mask = model.which_invalid(pars)
    pars[invalid_mask] = par_scales[invalid_mask]
    if not model.valid(pars):
        raise RuntimeError("Invalid guess could not be fixed. This should never happen.")

    # Compute the cost
    cost = np.sum((amplitudes - amplitudes_model) ** 2)
    return cost, pars
