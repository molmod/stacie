# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code)
when working with code in this repository.

## Overview

STACIE is a Python package for robust, uncertainty-aware estimation of autocorrelation integrals,
primarily used for transport properties in molecular dynamics.
Accuracy and statistical robustness take precedence over micro-optimizations.

Guidance that applies to only part of the repo lives next to the code it governs:

- `src/stacie/CLAUDE.md`: mathematical notation, Green-Kubo context, numerical stability, units.
- `tests/CLAUDE.md`: what a test has to prove and which helpers to use.
- `docs/CLAUDE.md`: documentation markup and the build scripts.

The release procedure is documented in `docs/source/development/release.md`.

## Non-Negotiables

- **Never run `docs/release_docs.sh`.**
  It checks out `gh-pages`, runs `git rm -rf .`, amends the single commit,
  and force-pushes to the remote.
  Deciding when to publish documentation is a human judgment call.
- **Never hardcode physical constants.**
  Use `scipy.constants`.
- **Never introduce a magic number as a reference value in a test.**
  Derive the expected result analytically,
  or take it from a well-known published reference,
  and say in the test where it comes from.
- **Uncertainty quantification must survive every change.**
  Robustness is STACIE's reason to exist,
  so a change that computes the right central value while dropping or corrupting the error bars
  is a regression, not an optimization.

## Commands

### Environment

The development environment is managed with [uv](https://docs.astral.sh/uv/), not with `pip`:

```bash
uv sync --extra=docs,tests,dev
```

### Linting

Pre-commit hooks run `ruff format`, `ruff check` and `markdownlint-cli2` automatically on commit.
After making changes, run all pre-commit checks before considering the work done:

```bash
pre-commit run --all
```

### Tests

```bash
pytest
```

Two consequences of `addopts` in `pyproject.toml` are worth knowing in advance:

- `-W error` turns every warning into a failure,
  so a new `RuntimeWarning` from an overflow or an invalid value fails the suite
  even when the assertions themselves pass.
- `-n auto --dist worksteal` already parallelizes the run through `pytest-xdist`.
  Never start a second `pytest` invocation while one is still running,
  because the doubled worker processes overload the machine.
  Run invocations one after another instead.

## Coding Conventions

### Semantic Line Breaks

All English text in this repo is wrapped using **semantic line breaks**:
break after sentences or logical units, not at a fixed character count.
This covers comments, docstrings, Markdown documentation, commit messages, and so on.
See <https://sembr.org/>.
Prose diffs then stay small, because editing one sentence never reflows its neighbours.

- **Every sentence starts on a new line.**
- **Break inside a sentence only where a break is needed, and then at a clause boundary.**
  A sentence that fits within the 100-character line length stays on a single line.
  A longer one is broken before a conjunction or a relative pronoun
  ("and", "but", "because", "which", "if", ...),
  or after a leading subordinate clause.
- **Not every comma is a break.**
  Enumerated items, appositions and short parentheticals stay on the line they started on.

The 100-character line length is a hard cap, not a target to fill.

### Avoid En and Em Dashes

Write sentences without en or em dashes.
They should never be used in any prose (code comments, docstrings, Markdown, ...),
neither in their UTF-8 glyph form nor in ASCII form (`--`, `---`).
Subclauses should be made explicit (e.g. "which", "because", "that")
or split into separate sentences.

### Prose That Ages Well

Stale prose is worse than no prose.
When writing comments, docstrings, or other prose, avoid:

- **Describing callers.**
  Do not note how other code uses a function or class.
  That is the caller's concern, and the remark silently rots when the caller changes.
- **Describing history.**
  Do not explain what the code used to do or how it changed.
  History belongs in commit messages and in `docs/source/development/changelog.md`.
- **Implementation details in docstrings.**
  Document the contract (how to use something), not how it works internally.
- **Line-number references.**
  They break as soon as the file changes.
  Point to a function, class, or file name instead.
- **Restating the code.**
  A comment should say something the code does not already say
  (the reason, the invariant, the non-obvious constraint),
  not paraphrase the next line.
  A purely redundant comment is not wrong, so nothing forces it to be updated,
  and it drifts out of sync silently.
- **Timeless phrasing for point-in-time claims.**
  An empirical observation about an external library or environment can stop being true
  after a version upgrade, with nothing to flag the comment as outdated.
  Say what was observed and, when it matters, on what
  (e.g. "as of SciPy 1.13, measured separately").

### Linting (ruff)

Do not add `# noqa` comments unless the violation is a genuine false positive
that cannot be resolved by restructuring the code,
because the `ignore` list in `pyproject.toml` already excludes the rules
that would fire spuriously in this codebase.

### Docstrings

Use **NumPy-style** sections (`Parameters`, `Returns`, `Raises`, `Notes`, ...),
rendered by Sphinx through `sphinx.ext.napoleon`.
Some conventions specific to this codebase:

- Docstrings are written in **reStructuredText**, not Markdown.
  This is the opposite of the Markdown used for documentation pages under `docs/source/`.
    - Use double backticks for inline code and parameter names.
    - Use Sphinx roles to cross-reference the API (e.g. ``:func:`estimate_acint` ``).
    - Write mathematical formulas in LaTeX.
- Lines are wrapped using semantic breaks, per [Semantic Line Breaks](#semantic-line-breaks) above.
- Use the imperative mood for function descriptions
  (e.g. "Compute the spectrum of the input sequences."),
  except for `@property` getters where the description should be a noun phrase
  (e.g. "The number of RFFT frequency grid points.").
- Do not repeat type annotations in the docstring,
  because they are already in the function signature.
- In `Returns` sections, use a **semantic name** for the return value, not the type:

    ```python
    # correct
    Returns
    -------
    result
        The inputs, intermediate results and outputs of the algorithm.

    # wrong, because the type is already in the signature
    Returns
    -------
    Result
        The inputs, intermediate results and outputs of the algorithm.
    ```

### Type Hints

All functions have type hints.
Use `numpy.typing.NDArray` or `numpy.typing.ArrayLike` for array arguments,
and document the expected shape in the docstring,
because the annotation cannot express it.

### Naming

Use descriptive names that reflect the underlying physics,
e.g. `acf_tail` instead of `temp_arr`.
Brevity matters too:
prefer the shortest name that still says what the quantity is.

### Data Classes

The project uses `attrs` for data classes, for as many classes as possible:
`@attrs.define`, fields declared with `attrs.field()`, and a docstring under each field.
Prefer this over `dataclasses`, `typing.NamedTuple` or a hand-written `__init__`.

### Dependencies

Runtime dependencies are declared in `pyproject.toml` under `[project] dependencies`.
Avoid adding heavy dependencies
unless they are strictly necessary for core scientific functionality.
Before adding a lazy import or a `try`/`except ImportError` guard,
check whether the package is already a declared dependency
and import it at the top of the file instead.

### Markdown

Section headings (`##`, `###`, ...) use **Title Case**
(capitalize nouns, verbs, adjectives, and adverbs;
lowercase articles, coordinating conjunctions, and prepositions regardless of length,
e.g. "from", "with").
Inline code spans keep their own casing and are never title-cased.
