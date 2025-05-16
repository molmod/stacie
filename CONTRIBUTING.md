# Contributor Guide

> This contributor guide is created with the following template:
> [nayafia/contributing-template](https://github.com/nayafia/contributing-template).

First of all, thank you for considering contributing to STACIE!
STACIE is being developed by academics who also have many other responsibilities,
and you are probably in a similar situation.
The purpose of this guide is to make efficient use of everyone's time.

STACIE has already been used for production simulations,
but we are always open to (suggestions for) improvements that fit within the goals of STACIE.
New worked examples that are not too computationally demanding are also highly appreciated!
Even simple things like correcting typos or fixing minor mistakes are welcome.

This section does not document how to use of Git and GitHub, or how to develop software in general.
We assume that you already have the basic skills to contribute.
Below are some links to documentation for those who are not familiar with these technicalities yet.

## Ground Rules

- We want everyone to have a positive experience with their (online) interactions related to
  STACIE's development.
  Our expectations for (online) communication are outlined in
  the [Code of Conduct](../code_of_conduct.md).

- Except for minor corrections, we encourage you to open a GitHub issue
  before making changes to the source code.
  A transparent discussion before making changes can save a lot of time.
  Also if you have found a potential problem but are not sure how to fix it,
  we encourage you to open an issue.

- When you contribute,
  you accept that your contributions will be distributed under the same
  [licenses](../getting_started/licenses.md) that
  we currently use for source code and documentation.

## How to Report a Bug

Create a new issue (or find an existing one) and include the following information:

1. What version of STACIE, Python and NumPy are you using?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

## First-Time Contributors

If you have never contributed to an open source project before,
you may find the following online references helpful:

- <http://makeapullrequest.com/>
- <http://www.firsttimersonly.com/>
- <https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github>

If something goes wrong in the process of creating a pull request, we'll try to help out.

## Contribution Workflow

Contributing to STACIE always involves the following steps:

1. Create an issue on GitHub to discuss your plans.
1. Fork the STACIE repository on GitHub.
1. Clone the original repository on your computer and add your fork as a second remote.
1. Install [pre-commit](https://pre-commit.com/).
1. Create a new branch. (Do not commit changes to the main branch.)
1. Make changes to the source code. New features must have unit tests and documentation.
1. Make sure all the tests pass, the documentation builds without errors or warnings,
   and pre-commit reports no problems.
1. Push your branch to your fork and Create a pull request on GitHub.
   In the pull request message (not the title), mention which issue the pull request addresses.
1. Wait for your changes to be reviewed
   and handle all requests for improvements during the review process.
1. If your change is accepted,
   it will be merged into the main branch and included in the next release of STACIE.
