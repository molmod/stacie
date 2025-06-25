# STACIE is a STable AutoCorrelation Integral Estimator.
# Copyright (C) 2024-2025 The contributors of the STACIE Python Package.
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
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import polygamma

__all__ = ("ExpPolyModel", "LorentzModel", "PadeModel", "SpectrumModel", "guess")


@attrs.define
class SpectrumModel:
    """Abstract base class for spectrum models.

    Subclasses must override all methods that raise ``NotImplementedError``.

    The first parameter must have a property that is used when constructing an initial guess:
    When the first parameter increases, the model should increase everywhere,
    and must allow for an arbitrary increase of the spectrum at all points.
    This is used to repair initial guesses that result in a partially negative spectrum.
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

    def valid(self, pars: NDArray[float]) -> bool:
        """Return ``True`` when the parameters are within the feasible region."""
        result = np.ones(pars.shape[:-1], dtype=bool)
        for i, (pmin, pmax) in enumerate(self.bounds()):
            result &= pmin < pars[..., i]
            result &= pars[..., i] < pmax
        return result

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
            "timestep": timestep,
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

    def solve_linear(
        self,
        freqs: NDArray[float],
        ndofs: NDArray[float],
        amplitudes: NDArray[float],
        weights: NDArray[float],
        nonlinear_pars: NDArray[float],
    ) -> NDArray[float]:
        """Use linear linear regression to solve a subset of the parameters.

        The default implementation in the base class assumes that the linear parameters
        are genuinely linear without rewriting the regression problem in a different form.

        Parameters
        ----------
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        amplitudes
            The amplitudes of the spectrum.
        ndofs
            The number of degrees of freedom at each frequency.
        weights
            Fitting weights, in range [0, 1], to use for each grid point.
        nonlinear_pars
            The values of the nonlinear parameters for which the basis functions are computed.

        Returns
        -------
        linear_pars
            The solved linear parameters.
        amplitudes_model
            The model amplitudes computed with the solved parameters.
        """
        nonlinear_pars = np.asarray(nonlinear_pars)
        if nonlinear_pars.ndim != 1:
            raise ValueError("The nonlinear parameters must be a 1D array.")
        nonlinear_mask = self.get_par_nonlinear()
        pars = np.ones(self.npar)
        pars[nonlinear_mask] = nonlinear_pars
        basis = self.compute(freqs, pars, deriv=1)[1][~nonlinear_mask]
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)
        rescaling = np.sqrt(weights) / amplitudes_std
        linear_pars = np.linalg.lstsq(
            (basis * rescaling).T,
            amplitudes * rescaling,
            # For compatibility with numpy < 2.0
            rcond=-1,
        )[0]
        amplitudes_model = np.dot(linear_pars, basis)
        return linear_pars, amplitudes_model

    def compute(
        self, freqs: NDArray[float], pars: NDArray[float], *, deriv: int = 0
    ) -> list[NDArray[float]]:
        """Compute the amplitudes of the spectrum model.

        Parameters
        ----------
        freqs
            The frequencies for which the model spectrum amplitudes are computed.
        pars
            The parameter vector.
            For vectorized calculations, the last axis corresponds to the parameter index.
        deriv
            The maximum order of derivatives to compute: 0, 1 or 2.

        Returns
        -------
        results
            A results list, index corresponds to order of derivative.
            The shape of the arrays in the results list is as follows:

            - For ``deriv=0``, the shape is ``(*vec_shape, len(freqs))``.
            - For ``deriv=1``, the shape is ``(*vec_shape, len(pars), len(freqs))``.
            - For ``deriv=2``, the shape is ``(*vec_shape, len(pars), len(pars), len(freqs))``

            If some derivatives are independent of the parameters,
            broadcasting rules may be used to reduce the memory footprint.
            This means that ``vec_shape`` may be replaced by a tuple of ones with the same length.
        """
        raise NotImplementedError

    def neglog_prior(self, pars: NDArray[float], *, deriv: int = 0) -> list[NDArray[float]]:
        """Minus logarithm of the prior probability density function, if any.

        Subclasses may implement (a very weak) prior, if any.
        """
        vec_shape = pars.shape[:-1]
        par_shape = pars.shape[-1:]
        result = [np.zeros(vec_shape)]
        if deriv > 0:
            result.append(np.zeros(vec_shape + par_shape))
        if deriv > 1:
            result.append(np.zeros(vec_shape + par_shape + par_shape))
        if deriv > 2:
            raise ValueError("Third or higher derivatives are not supported.")
        return result

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return autocorrelation integral (and other properties) derived from the parameters.

        Parameters
        ----------
        pars
            The parameters.
        covar
            The covariance matrix of the parameters.

        Returns
        -------
        props
            A dictionary with additional properties,
            whose calculation requires model-specific knowledge.
        """
        return {}


def _convert_degrees(value: ArrayLike) -> NDArray[int]:
    """Convert the input to an array of integers."""
    value = np.asarray(value, dtype=int)
    if value.ndim == 0:
        return value[None]
    if value.ndim > 1:
        raise ValueError("Input must be a scalar or a 1D array.")
    value.sort()
    return value


def _validate_degrees(obj, attribute, value: NDArray[int]) -> None:
    """Validate the degrees of the polynomial."""
    if not np.all(value >= 0):
        raise ValueError("All degrees must be non-negative.")
    if len(value) == 0:
        raise ValueError("The list of degrees must not be empty.")
    if len(value) != len(set(value)):
        raise ValueError("The list of degrees must not contain duplicates.")


@attrs.define
class ExpPolyModel(SpectrumModel):
    """Exponential function of a linear combination of simple monomials."""

    degrees: NDArray[int] = attrs.field(converter=_convert_degrees, validator=_validate_degrees)
    """The degree of the monomials."""

    @property
    def name(self):
        degrees_str = ", ".join(map(str, self.degrees))
        return f"exppoly({degrees_str})"

    def bounds(self) -> list[tuple[float, float]]:
        """Return parameter bounds for the optimizer."""
        return [(-np.inf, np.inf)] * self.npar

    @property
    def npar(self):
        """Return the number of parameters."""
        return len(self.degrees)

    @property
    def par_scales(self) -> NDArray[float]:
        """Return the scales of the parameters and the cost function."""
        return self.scales["freq_scale"] ** -self.degrees

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters."""
        return np.zeros(self.npar, dtype=bool)

    def solve_linear(
        self,
        freqs: NDArray[float],
        ndofs: NDArray[float],
        amplitudes: NDArray[float],
        weights: NDArray[float],
        nonlinear_pars: NDArray[float],
    ) -> NDArray[float]:
        """Use linear linear regression to solve a subset of the parameters.

        This is a specialized implementation that rewrites the problem
        in a different form to solve all parameters with a linear regression.
        """
        if len(nonlinear_pars) != 0:
            raise ValueError("The number of nonlinear parameters must be exactly 0.")
        log_amplitudes = np.log(amplitudes)
        log_amplitudes_std = polygamma(1, 0.5 * ndofs)
        rescaling = np.sqrt(weights) / log_amplitudes_std
        expected_values = log_amplitudes * rescaling
        design_matrix = np.power.outer(freqs, self.degrees) * rescaling[:, None]
        pars = np.linalg.lstsq(design_matrix, expected_values, rcond=-1)[0]
        amplitudes_model = np.exp(np.dot(design_matrix, pars))
        return pars, amplitudes_model

    def compute(
        self, freqs: NDArray[float], pars: NDArray[float], *, deriv: int = 0
    ) -> list[NDArray[float]]:
        """See :py:meth:`SpectrumModel.compute`."""
        if not isinstance(deriv, int):
            raise TypeError("Argument deriv must be integer.")
        if deriv < 0:
            raise ValueError("Argument deriv must be zero or positive.")
        if freqs.ndim != 1:
            raise TypeError("Argument freqs must be a 1D array.")

        # Construct a basis of simple monomials.
        basis = np.power.outer(freqs, self.degrees).T

        # Compute model amplitudes and derivatives.
        func = np.exp(np.einsum("...p,pf->...f", pars, basis))
        results = [func]
        if deriv >= 1:
            results.append(np.einsum("...f,pf->...pf", func, basis))
        if deriv >= 2:
            results.append(np.einsum("...f,pf,qf->...pqf", func, basis, basis))
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return autocorrelation integral (and other properties) derived from the parameters."""
        # The logarithm of the autocorrelation integral is the first parameter,
        # which is assumed to be normally distributed.
        log_acint = pars[0]
        log_acint_var = covar[0, 0]
        # The autocorrelation integral is the exponential of the first parameter,
        # and is therefore log-normally distributed.
        acint = np.exp(log_acint + 0.5 * log_acint_var)
        acint_var = (np.exp(log_acint_var) - 1) * np.exp(2 * log_acint + log_acint_var)
        return {
            "log_acint": log_acint,
            "log_acint_var": log_acint_var,
            "acint": acint,
            "acint_var": acint_var,
        }


@attrs.define
class PadeModel(SpectrumModel):
    """A rational function model for the spectrum, a.k.a. a Padé approximation."""

    numer_degrees: NDArray[int] = attrs.field(
        converter=_convert_degrees, validator=_validate_degrees
    )
    """The degrees of the monomials in the numerator."""

    denom_degrees: NDArray[int] = attrs.field(
        converter=_convert_degrees, validator=_validate_degrees
    )
    """The degrees of the monomials in the denominator.

    Note that the leading term is always 1, and there is no need to include
    degree zero.
    """

    @property
    def name(self):
        numer_str = ", ".join(map(str, self.numer_degrees))
        denom_str = ", ".join(map(str, self.denom_degrees))
        return f"pade({numer_str}; {denom_str})"

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
                self.scales["amp_scale"]
                * self.scales["freq_scale"] ** (-np.array(self.numer_degrees)),
                self.scales["freq_scale"] ** (-np.array(self.denom_degrees)),
            ]
        )

    def get_par_nonlinear(self) -> NDArray[bool]:
        """Return a boolean mask for the nonlinear parameters."""
        return np.zeros(self.npar, dtype=bool)

    def solve_linear(
        self,
        freqs: NDArray[float],
        ndofs: NDArray[float],
        amplitudes: NDArray[float],
        weights: NDArray[float],
        nonlinear_pars: NDArray[float],
    ) -> NDArray[float]:
        """Use linear linear regression to solve a subset of the parameters.

        This is a specialized implementation that rewrites the problem
        in a different form to solve all parameters with a linear regression.
        """
        if len(nonlinear_pars) != 0:
            raise ValueError("The number of nonlinear parameters must be exactly 0.")
        # Rescale frequencies as a simple form of preconditioning.
        x = freqs / freqs[-1]

        # Construct bases of monomials.
        basis_n = np.power.outer(x, self.numer_degrees).T
        basis_d = np.power.outer(x, self.denom_degrees).T

        # Set up and solve linear regression problem.
        amplitudes_std = amplitudes / np.sqrt(0.5 * ndofs)
        rescaling = np.sqrt(weights) / amplitudes_std
        expected_values = amplitudes * rescaling
        part_n = basis_n * rescaling
        part_d = -basis_d * expected_values
        design_matrix = np.concatenate([part_n, part_d]).T
        pars = np.linalg.lstsq(design_matrix, expected_values, rcond=-1)[0]

        # Compute fitted model amplitudes.
        npar_n = len(self.numer_degrees)
        pars_n = pars[:npar_n]
        pars_d = pars[npar_n:]
        amplitudes_model = np.dot(pars_n, basis_n) / (1 + np.dot(pars_d, basis_d))

        # Convert parameters to the original scale.
        pars = np.zeros(self.npar)
        pars[:npar_n] = pars_n / self.scales["freq_scale"] ** self.numer_degrees
        pars[npar_n:] = pars_d / self.scales["freq_scale"] ** self.denom_degrees

        return pars, amplitudes_model

    def compute(
        self, freqs: NDArray[float], pars: NDArray[float], *, deriv: int = 0
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
        pars = np.asarray(pars)
        if pars.shape[-1] != npar_n + npar_d:
            raise ValueError("The number of parameters does not match the model.")

        # Construct two bases of monomials.
        basis_n = np.power.outer(freqs, self.numer_degrees).T
        basis_d = np.power.outer(freqs, self.denom_degrees).T
        pars_n = pars[..., :npar_n]
        pars_d = pars[..., npar_n:]

        # Compute model amplitudes and derivatives.
        num = np.einsum("...p,pf->...f", pars_n, basis_n)
        denom = 1 + np.einsum("...p,pf->...f", pars_d, basis_d)
        results = [num / denom]
        vec_shape = pars.shape[:-1]
        if deriv >= 1:
            model_grad = np.empty((*vec_shape, npar_n + npar_d, *freqs.shape))
            np.einsum("pf,...f->...pf", basis_n, 1 / denom, out=model_grad[..., :npar_n, :])
            block_n = model_grad[..., :npar_n, :]
            np.einsum(
                "pf,...f->...pf", basis_d, -results[0] / denom, out=model_grad[..., npar_n:, :]
            )
            results.append(model_grad)
        if deriv >= 2:
            model_hess = np.zeros((*vec_shape, npar_n + npar_d, npar_n + npar_d, *freqs.shape))
            np.einsum(
                "...pf,...f,qf->...pqf",
                block_n,
                1 / denom,
                -basis_d,
                out=model_hess[..., :npar_n, npar_n:, :],
            )
            np.einsum(
                "...pqf->...qpf",
                model_hess[..., :npar_n, npar_n:, :],
                out=model_hess[..., npar_n:, :npar_n, :],
            )
            np.einsum(
                "...f,pf,qf->...pqf",
                2 * results[0] / denom**2,
                basis_d,
                basis_d,
                out=model_hess[..., npar_n:, npar_n:, :],
            )
            results.append(model_hess)
        if deriv >= 3:
            raise ValueError("Third or higher derivatives are not supported.")
        return results

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return autocorrelation integral (and other properties) derived from the parameters."""
        return {
            "acint": pars[0],
            "acint_var": covar[0, 0],
        }


@attrs.define
class LorentzModel(PadeModel):
    """A model for the spectrum with a Lorentzian peak at zero frequency plus some white noise.

    This is a special case of the PadeModel with
    `numer_degrees` = [0, 2] and `denom_degrees` = [2].
    Furthermore, it will only accept parameters that correspond
    to a well-defined exponential correlation time.
    """

    relerr_threshold: float = attrs.field(default=0.1)
    """A threshold for the relative error on the exponential correlation time.

    The relative error is defined as ratio of the standard deviation
    to the mean of the correlation time.

    To accept the current cutoff frequency, the relative error must be below this threshold.
    This eliminates high-variance cases for which the maximum a posteriori estimate
    tends to be a poor approximation.
    """

    numer_degrees: NDArray[int] = attrs.field(init=False, factory=lambda: np.array([0, 2]))
    denom_degrees: NDArray[int] = attrs.field(init=False, factory=lambda: np.array([2]))

    @property
    def name(self):
        return f"lorentz({self.relerr_threshold})"

    def derive_props(
        self, pars: NDArray[float], covar: NDArray[float]
    ) -> dict[str, NDArray[float]]:
        """Return autocorrelation integral (and other properties) derived from the parameters.

        The exponential correlation time is derived from the parameters, if possible.
        If not, or if the variance of the estimate is too large,
        the "criterion" is set to infinity and the "msg" is set accordingly,
        to discard the current fit from the average over the cutoff frequencies.
        """
        result = super().derive_props(pars, covar)
        # Try to deduce the exponential correlation time from the parameters.
        if (
            # real correlation time
            pars[2] > 0
            # positive lorentzian amplitude, maximum in the origin
            and pars[0] * pars[2] > pars[1]
        ):
            # The following estimates of the exponential correlation time and its variance
            # are only valid for small variances.
            tau = np.sqrt(pars[2]) / (2 * np.pi)
            dtau_dp = 1 / (4 * np.pi * np.sqrt(pars[2]))
            tau_var = covar[2, 2] * dtau_dp**2
            tau_props = {
                "corrtime_exp": tau,
                "corrtime_exp_var": tau_var,
                "exp_block_time": tau * np.pi / 10,
                "exp_block_time_var": tau_var * np.pi / 10,
                "exp_simulation_time": 20 * tau * np.pi,
                "exp_simulation_time_var": 20 * tau_var * np.pi,
            }
            result.update(tau_props)
        # If we fail to find the exponential correlation time with a decent relative error,
        # we discard the entire estimate at this cutoff frequency.
        if "corrtime_exp" not in result:
            result["criterion"] = np.inf
            result["msg"] = "No correlation time estimate available."
        elif result["corrtime_exp_var"] ** 0.5 > result["corrtime_exp"] * self.relerr_threshold:
            result["criterion"] = np.inf
            result["msg"] = "Variance of the correlation time estimate is too large."
        return result


def guess(
    freqs: NDArray[float],
    ndofs: NDArray[float],
    amplitudes: NDArray[float],
    weights: NDArray[float],
    model: SpectrumModel,
    rng: np.random.Generator,
    nonlinear_budget: int,
):
    """Guess initial values of the parameters for a model.

    Parameters
    ----------
    freqs
        The frequencies for which the model spectrum amplitudes are computed.
    ndofs
        The number of degrees of freedom at each frequency.
    amplitudes
        The amplitudes of the spectrum.
    weights
        Fitting weights, in range [0, 1], to use for each grid point.
    model
        The model for which the parameters are guessed.
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
        return _guess_linear(freqs, ndofs, amplitudes, weights, model, [], nonlinear_mask)[1]

    # Otherwise, we need to sample the nonlinear parameters and guess the linear parameters.
    nonlinear_samples = model.sample_nonlinear_pars(rng, nonlinear_budget**num_nonlinear_pars)
    best = None
    for nonlinear_pars in nonlinear_samples:
        cost, pars = _guess_linear(
            freqs, ndofs, amplitudes, weights, model, nonlinear_pars, nonlinear_mask
        )
        if best is None or best[0] > cost:
            best = cost, pars
    return best[1]


def _guess_linear(
    freqs: NDArray[float],
    ndofs: NDArray[float],
    amplitudes: NDArray[float],
    weights: NDArray[float],
    model: SpectrumModel,
    nonlinear_pars: NDArray[float],
    nonlinear_mask: NDArray[bool],
) -> tuple[float, NDArray[float]]:
    """Guess initial values of the linear parameters for a model.

    Parameters
    ----------
    freqs
        The frequencies for which the model spectrum amplitudes are computed.
    ndofs
        The number of degrees of freedom at each frequency.
    amplitudes
        The amplitudes of the spectrum.
    weights
        Fitting weights, in range [0, 1], to use for each grid point.
    model
        The model for which the parameters are guessed.
    nonlinear_pars
        The values of the nonlinear parameters.
    nonlinear_mask
        A boolean mask for the nonlinear parameters.

    Returns
    -------
    cost
        The cost of the guess.
    pars
        An initial guess of the parameters.
    """
    # Perform a weighted least squares fit to guess the linear parameters.
    linear_pars, amplitudes_model = model.solve_linear(
        freqs, ndofs, amplitudes, weights, nonlinear_pars
    )

    # Combine the linear and nonlinear parameters
    pars = np.zeros(model.npar)
    pars[nonlinear_mask] = nonlinear_pars
    pars[~nonlinear_mask] = linear_pars

    # Fix invalid guesses
    invalid_mask = model.which_invalid(pars)
    par_scales = model.par_scales
    pars[invalid_mask] = par_scales[invalid_mask]
    if not model.valid(pars):
        raise RuntimeError("Invalid guess could not be fixed. This should never happen.")

    # If the model has zero or negative values,
    # increase the value of the constant term to fix the guess.
    def negerr(x):
        """Returns a negative value if the spectrum is less than 1e-3 of ``amp_scale``.

        Parameters
        ----------
        x
            A correction to the first element of the parameter vector.
        """
        mod_pars = pars.copy()
        mod_pars[0] += x
        return model.compute(freqs, mod_pars)[0].min() - model.scales["amp_scale"] * 1e-3

    if negerr(0) < 0.0:
        bracket = _find_bracket(negerr, 0, par_scales[0])
        xopt = brentq(negerr, bracket[0], bracket[1])
        pars[0] += xopt
        amplitudes_model = model.compute(freqs, pars)[0]

    # Compute the cost
    delta = amplitudes - amplitudes_model
    cost = np.einsum("i,i,i", delta, delta, weights)
    return cost, pars


def _find_bracket(f, x0, x1):
    """Fit a bracket with a sign change.

    This function assumes that a sign change is present either in [x0, x1]
    or outside the bracket on the side of x1.
    """
    f0 = f(x0)
    f1 = f(x1)
    for _ in range(100):
        if np.sign(f0) != np.sign(f1):
            break
        x2 = x1 + (x1 - x0) * 2
        f2 = f(x2)
        x0, x1 = x1, x2
        f0, f1 = f1, f2
    else:
        raise RuntimeError("Failed to find a bracket with a sign change.")
    return x0, x1
