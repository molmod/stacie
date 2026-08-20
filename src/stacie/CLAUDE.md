# CLAUDE.md (Core Library)

Guidance for the STACIE library code itself.
The general conventions in the repository root `CLAUDE.md` apply here too.

## Scientific Validity

Changes to this code change physical results, so the following are correctness requirements,
not stylistic preferences.

### Mathematical Notation

Distinguish clearly between sampling averages and expectation values.
In LaTeX inside docstrings, a sampling average carries a hat (`\hat{x}`)
and the corresponding expectation value does not (`x`).
Blurring the two in a docstring is a real error,
because a reader then cannot tell whether a quantity carries sampling noise.

### The Green-Kubo Context

Integrals of autocorrelation functions relate directly to physical properties
such as viscosity and diffusivity.
Code changes must not violate the underlying physics.
When a change alters what is integrated, over which range, or with which prefactor,
say in the docstring which physical quantity the result now corresponds to.

### Numerical Stability

Use numerically stable algorithms.
Pre-condition fits rather than relying on the optimizer to cope with badly scaled parameters,
and be mindful of the propagation of truncation and rounding errors.
Note that `-W error` in the test configuration turns an overflow or an invalid value
into a test failure, so a numerically fragile path fails loudly rather than silently.

### Units and Dimensions

STACIE is unit agnostic, with one exception that has to be tracked explicitly:
the time step has a dimension of time,
and it determines the unit of the final autocorrelation integral.
Document dimensions in docstrings wherever a quantity is not dimensionless,
because the code alone cannot express them.

## `__all__`

Wildcard imports are banned (ruff `F403`),
so `__all__` does not describe a star-import surface here.
It is the module's **import contract**: the names that code outside the module is meant to import.

- Every module here declares `__all__` directly after the imports.
  It is a tuple of string literals, sorted (enforced by ruff `RUF022`).
- List a name when it is imported by another `stacie` module,
  by a user's analysis script, or by the documentation.
- Do not list module-internal helpers, even when they lack a leading underscore.
  A public-looking name is not a claim that the name is exported.
- Tests may import names that are not in `__all__`;
  white-box testing does not make a name part of the contract.
- Do not re-export:
  a name in `__all__` must be defined in that same module.
  Import a name from the module that defines it,
  not from a module that happens to import it.

## Published Algorithms

STACIE's algorithms are described in:

> Gözdenur Toraman, Dieter Fauconnier, and Toon Verstraelen,
> "STable AutoCorrelation Integral Estimator:
> Robust and Accurate Transport Properties from Molecular Dynamics Simulations",
> *Journal of Chemical Information and Modeling* 2025, 65 (19), 10445-10464,
> <https://doi.org/10.1021/acs.jcim.5c01475>

The development version may have evolved beyond the description in the paper,
so the paper is authoritative for the core principles
but not necessarily for the current signatures or defaults.
When the code and the paper disagree on a detail, the code is what ships;
check whether the difference is intentional before treating either one as a bug.
