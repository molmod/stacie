# CLAUDE.md (Tests)

Guidance for the test suite.
The general conventions in the repository root `CLAUDE.md` apply here too.

## What a Test Has to Prove

Every new feature comes with tests.
A test that only pins down the current output is worth little here,
because it passes just as happily when the current output is wrong.
Prefer, in this order:

1. **An analytical reference.**
   Derive the expected value by hand and state the derivation in a comment or docstring.
2. **A published reference.**
   Cite where the number comes from.
3. **A consistency test against a naive implementation.**
   Write the simple, obviously correct, inefficient version inside the test
   and compare it to STACIE's optimized code path.
   This catches optimizations that quietly change the result.

Never paste a number from a previous run as the expected value.
See the Non-Negotiables in the root `CLAUDE.md`.

## Edge Cases

Time-series estimators fail at the margins, so test the margins:
very short time series, poorly sampled data, a single independent sequence,
spectra with the DC component removed, and inputs where the fit cannot converge.
A feature that only works on comfortable data is not finished.

## Derivatives

Hand-coded derivatives are verified numerically with `numdifftools`.
Use the helpers in `conftest.py` rather than calling `numdifftools` directly:
`check_deriv`, `check_curv`, `check_gradient` and `check_hessian`.
They already implement STACIE's convention
that a function returns its value and derivatives as a tuple selected by the `deriv` argument,
and they scale the comparison by the numerical error estimate,
so a plain absolute tolerance is not needed and should not be reintroduced.

## Regression Outputs

`test_regression.py` uses `pytest-regtest`, with recorded output under `tests/_regtest_outputs/`.
These files are generated, so do not hand-edit them.
When a change is genuinely supposed to alter the recorded output,
regenerate them with `pytest --regtest-reset`
and review the resulting diff as carefully as the code change itself,
because that diff is the only evidence of what the change did to the numbers.

## Warnings Are Errors

`-W error` is set in `pyproject.toml`.
A new warning from NumPy, SciPy or Matplotlib fails the suite.
Do not silence one with a blanket `filterwarnings` entry;
fix the underlying overflow, invalid value or deprecated call instead.
