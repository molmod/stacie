<!-- markdownlint-disable no-duplicate-heading blanks-around-headings -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Effort-based Versioning](https://jacobtomlinson.dev/effver/).

## [Unreleased]

### Changed

- Switch to a choice of license (`CC-BY-SA-4.0 OR LGPL-3.0-or-later`) for the documentation only.
  STACIE's code is still distributed under `LGPL-3.0-or-later`.

### Fixed

- Several documentation improvements:
    - Clarify how the derivation of block-averaged velocities for diffusion and electrical conductivity,
      using atomic positions (or dipole vectors) as input.
    - Improve explanation on discarding the DC-component of the spectrum.
    - Add more helpful comments on how to deal with unit conversion.
    - Fixed a typo in the equation of the marginalization weights.
- Repocard images were added.
- Dataset metadata improvements.
- Several other minor issues in documentation and tooling were fixed.

(v1.0.0)=
## [1.0.0] - 2025-06-26

This is the first stable release of STACIE!

### Changed

- Metadata and citation updates

(v1.0.0rc1)=
## [1.0.0rc1] - 2025-06-25

This is the first release candidate of STACIE, with a final release expected very soon.
The main remaining issues are related to (back)linking of external resources
in the documentation and README files.

[Unreleased]: https://github.com/molmod/stacie
[1.0.0]: https://github.com/molmod/stacie/releases/tag/v1.0.0
[1.0.0rc1]: https://github.com/molmod/stacie/releases/tag/v1.0.0rc1
