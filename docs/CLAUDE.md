# CLAUDE.md (Documentation)

Guidance for the documentation sources and build.
The general conventions in the repository root `CLAUDE.md` apply here too.

## Markup

Documentation pages under `source/` are written in **MyST Markdown**, not reStructuredText.
This is the opposite of the docstrings in `src/stacie/`, which are reStructuredText.
When moving text between a docstring and a page, the inline markup has to be converted:
double backticks become single backticks,
and Sphinx roles become MyST equivalents.

Semantic line breaks apply to every Markdown file here.

## Building

Run the scripts from within this directory:

```bash
./compile_html.sh     # sphinx-build -M html, with -W --keep-going --nitpicky
./compile_pdf.sh      # regenerates the logo PDF with inkscape, then latexpdf
./clean.sh            # removes build/ and jupyter_execute/
```

`-W --nitpicky` means a broken cross-reference or an unresolvable type annotation
fails the build rather than producing a warning,
so a documentation build failure is often really an API docstring problem.

For iterative work, `./preview_html.sh` runs `sphinx-autobuild`
and watches both `source/` and `../src/stacie/`,
so docstring edits refresh the preview too.

Never run `./release_docs.sh`.
See the Non-Negotiables in the root `CLAUDE.md`.

## Executable Content

Worked examples under `source/examples/` are plain `.py` files, not `.ipynb`.
They are written in jupytext `py:percent` format
and turned into notebooks at build time by `nb_custom_formats` in `conf.py`,
then executed by `myst-nb`.
This is why they are readable in a diff and why they carry no stored output.

Consequences worth knowing before editing one:

- Cell boundaries are `# %%` comments, and Markdown cells are `# %% [markdown]`.
  Breaking that structure silently changes how the page renders.
- Execution is cached (`nb_execution_mode = "cache"`), with a 300 second timeout per notebook.
  Keep new examples computationally cheap, because they run in CI as well.
- `source/examples/` has its own `ruff.toml`,
  so the lint rules there differ from the rest of the repo on purpose.

## Changelog

Describe user-visible changes in `source/development/changelog.md`.
The entry says what changed for someone using STACIE,
not which functions were touched.
