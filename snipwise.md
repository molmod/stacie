# Snipwise Configuration

Consult the full documentation at <https://reproducible-reporting.github.io/snipwise/>.

```toml
# The BibTeX records of the two papers, which are entire files in the documentation.
[[sources]]
patterns = ["docs/source/getting_started/reference_stacie.bib"]
scanner = "whole"
snippets = ["stacie-ref-bib"]

[[sources]]
patterns = ["docs/source/getting_started/reference_lorentz.bib"]
scanner = "whole"
snippets = ["lorentz-ref-bib"]

# The README quotes both records as fenced BibTeX inside a block quote.
[[targets]]
patterns = ["README.md"]
scanner = "markers"
render = '{{ ("\n" + (content | codeblock("bibtex")) + "\n\n") | prefix("> ") }}'
snippets = ["stacie-ref-bib", "lorentz-ref-bib"]

# The Markdown files have several snippets
[[targets]]
patterns = ["README.md", "docs/source/index.md", "docs/source/getting_started/cite.md"]
scanner = "markers"

# The abstract of the citation metadata, which is a folded YAML block scalar.
# The template terminates the last line, because the region includes its newline.
[[targets]]
patterns = ["CITATION.cff"]
scanner = "regex"
regex = '(?m)^abstract: >-\n(?P<content>(?:^  .*\n)+)'
snippets = ["abstract"]
render = "{{ content | plain | prefix('  ') }}\n"

# The summary of the Python package metadata, which is a single-line TOML string.
[[targets]]
patterns = ["pyproject.toml"]
scanner = "regex"
regex = '(?m)^description = "(?P<content>[^"]*)"$'
snippets = ["tagline"]
render = "{{ content | unwrap }}"

# The docstring of the package, which is a single-line string.
[[targets]]
patterns = ["src/stacie/__init__.py"]
scanner = "regex"
regex = '(?m)^"""(?P<content>[^"]*)"""$'
snippets = ["tagline"]
render = "{{ content | unwrap }}"

# The keywords array of the Python package metadata.
[[targets]]
patterns = ["pyproject.toml"]
scanner = "markers"
snippets = ["keywords"]
render = '''{{ content | prefix('"') | suffix('",') }}'''

# The keywords sequence of the citation metadata.
[[targets]]
patterns = ["CITATION.cff"]
scanner = "markers"
snippets = ["keywords"]
render = "{{ content | prefix('- ') }}"
```

## `tagline`

```text
STACIE is the STable AutoCorrelation Integral Estimator for time-correlated data.
```

## `abstract`

```markdown
STACIE is a Python package and algorithm that computes time integrals of autocorrelation functions.
It is primarily designed for post-processing molecular dynamics simulations.
However, it can also be used for more general analysis of time-correlated data.
Typical applications include estimating transport properties
and the uncertainty of averages over time-correlated data,
as well as analyzing characteristic timescales.
```

## `keywords`

```text
ACF
autocorrelation function
autocorrelation integral
characteristic timescales
data analysis
exponential correlation time
integrated correlation time
molecular dynamics
open source
post-processing
power spectral distribution
PSD
Python package
scientific computing
STACIE
time correlation
transport properties
uncertainty quantification
```
